import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snip_search as search
import train_snip_compact as train
import single_filter_lora as sf
from test_peft_integration import CASES, ADAPTERS, expected_adapter_ids
from test_peft_training import tiny_workflow


def tiny():
    torch.set_num_threads(2)
    return VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                             depth=3, num_heads=2, num_classes=3)


def dataset(n=64):
    g = torch.Generator().manual_seed(42)
    return TensorDataset(torch.randn(n, 3, 32, 32, generator=g),
                         torch.randint(0, 3, (n,), generator=g))


def arguments(case, output, filter_type="symbolic"):
    kind, tuner, init = case
    return train.parser().parse_args([
        "--device", "cpu", "--epochs", "2", "--output-dir", str(output),
        "--filter-type", filter_type, "--filter-rank", "2", "--symbolic-search-tokens", "32",
        "--adapter-type", kind, "--paca-tuner", tuner, "--init-method", init,
        "--lora-rank", "2", "--lora-alpha", "4", "--paca-rank", "3",
        "--paca-adapter-rank", "2", "--snip-batch-size", "16"])


def test_scores_match_full_calibration_loss_and_restore_state():
    model = tiny(); data = dataset(7)
    model.eval()
    params = list(model.parameters())
    loss = nn.functional.cross_entropy(model(data.tensors[0]), data.tensors[1])
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    expected = {id(p): float((p.detach()*g).abs().double().sum()) if g is not None else 0
                for p, g in zip(params, grads)}
    for i, p in enumerate(params):
        p.requires_grad_(i % 2 == 0)
        p.grad = torch.ones_like(p)
    model.train(); model.blocks[1].eval()
    flags = [p.requires_grad for p in params]
    modes = [m.training for m in model.modules()]
    weights = [p.detach().clone() for p in params]
    actual = search.snip_scores(model, DataLoader(data, batch_size=3), "cpu")
    assert actual["samples"] == 7
    assert actual["network_score"] == pytest.approx(sum(expected.values()), rel=1e-5)
    for i, block in enumerate(model.blocks):
        assert actual["block_scores"][i] == pytest.approx(sum(expected[id(p)] for p in block.parameters()), rel=1e-5)
    assert [p.requires_grad for p in params] == flags
    assert [m.training for m in model.modules()] == modes
    assert all(torch.equal(p, old) for p, old in zip(params, weights))
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in params)


def test_empty_score_loader_restores_flags():
    model = tiny(); sf.freeze_non_trainable(model, [])
    flags = [p.requires_grad for p in model.parameters()]
    with pytest.raises(ValueError, match="Empty"):
        search.snip_scores(model, DataLoader(dataset(0)), "cpu")
    assert [p.requires_grad for p in model.parameters()] == flags


def test_highest_score_excludes_replacement_and_breaks_ties():
    assert search.select_sensitive_block({0: 90, 1: 4, 2: 4}, 0) == 1
    with pytest.raises(ValueError):
        search.select_sensitive_block({0: 90}, 0)


@pytest.mark.parametrize("case", [case for case in CASES if not case[0].startswith("uni")])
@pytest.mark.parametrize("filter_type", ["lowrank", "symbolic"])
def test_training_search_scope_and_checkpoint(case, filter_type, tmp_path, monkeypatch):
    original = tiny(); saved = copy.deepcopy(original.state_dict()); models = []
    def factory(*a, **k):
        model = copy.deepcopy(original)
        models.append(model)
        return model
    def loaders(args):
        return DataLoader(dataset(), batch_size=23, shuffle=True,
                          generator=torch.Generator().manual_seed(args.seed)), \
            DataLoader(dataset(8), batch_size=8), DataLoader(dataset(8), batch_size=8), 3, ["a", "b", "c"]
    monkeypatch.setattr(train.timm, "create_model", factory)
    monkeypatch.setattr(train, "get_dataloaders", loaders)
    monkeypatch.setattr(train, "plot_training_curves", lambda *a, **k: {})
    monkeypatch.setattr(train, "plot_lr_schedule", lambda *a, **k: "omitted")
    monkeypatch.setattr(train, "plot_param_breakdown", lambda *a, **k: "omitted")
    optimizer_steps = []
    original_snapshot = search.cpu_optimizer_state
    def checked_snapshot(optimizer):
        state = original_snapshot(optimizer)
        optimizer_steps.append(max(float(value["step"]) for value in state["state"].values()))
        assert all(not value.is_cuda for row in state["state"].values()
                   for value in row.values() if torch.is_tensor(value))
        return state
    monkeypatch.setattr(search, "cpu_optimizer_state", checked_snapshot)
    args = arguments(case, tmp_path, filter_type)
    path = train.train(args)
    summary = json.loads(path.read_text())
    selection = summary["snip_selection"]
    replaced, sensitive = summary["replaced_block_idx"], summary["peft_block_idx"]
    assert replaced != sensitive
    assert selection["search_complete"] and selection["reused_winning_candidate"]
    assert selection["optimizer_continuity"] and selection["init_images"] == 64
    assert selection["scoring_images"] == 64
    scores = {int(i): score for i, score in selection["target_scores"]["block_scores"].items()}
    assert sensitive == search.select_sensitive_block(scores, replaced)
    averages = {}
    alive = [0, 1, 2]
    for epoch in selection["search_history"]:
        rows = epoch["candidates"]
        assert {row["candidate_block"] for row in rows} == set(alive)
        assert {row["train_samples"] for row in rows} == {64}
        assert {row["optimizer_steps"] for row in rows} == {3}
        for row in rows:
            i = row["candidate_block"]
            expected = .9 * averages.get(i, 0) + .1 * row["network_snip"]
            assert row["ema"] == pytest.approx(expected)
            averages[i] = expected
        ordered = sorted(alive, key=lambda i: (-averages[i], i))
        assert epoch["survived"] == ordered[:(len(alive)+1)//2]
        alive = epoch["survived"]
    assert alive == [replaced]
    assert optimizer_steps == [3, 3, 3, 6, 6]
    model = train.restore_model(summary, "cpu")
    wrappers = [m for m in model.modules() if isinstance(m, ADAPTERS)]
    assert len(wrappers) == 4
    assert all(any(m is w for m in model.blocks[sensitive].modules()) for w in wrappers)
    expected = expected_adapter_ids(model)
    expected.update(id(p) for p in model.blocks[replaced].parameters())
    expected.update(id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters())
    expected.update(id(p) for p in model.head.parameters())
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected
    assert summary["param_breakdown"]["trainable_backbone"] == 0
    assert summary["epochs_trained"] == 2 and summary["test_evaluated"]
    assert summary["final_test_acc"] is not None
    # All untouched dense blocks preserve their exact pretrained weights through search and training.
    for i in set(range(3)) - {replaced, sensitive}:
        for name, p in model.blocks[i].named_parameters():
            if not p.requires_grad:
                assert torch.equal(p, saved[f"blocks.{i}.{name}"])
    # Repeated reconstruction must preserve symbolic operators and PaCA column identities.
    again = train.restore_model(summary, "cpu")
    x = dataset(2).tensors[0]
    assert torch.allclose(model(x), again(x), atol=1e-6)
    result = train.evaluate_summary(path, "cpu")
    assert result["final_test_acc"] == summary["final_test_acc"]
    assert (path.parent / "snip_block_scores.csv").is_file()
    assert (path.parent / "filter_equations.json").is_file()


@pytest.mark.parametrize("extra", [["--snip-search-epochs", "1"],
    ["--snip-momentum", "nan"], ["--snip-prune-fraction", "0"],
    ["--snip-samples", "0"], ["--epochs", "0"],
    ["--adapter-type", "paca", "--init-method", "loftq"],
    ["--adapter-type", "paca", "--lora-ortho-lambda1", "1"]])
def test_invalid_config_rejected(extra):
    with pytest.raises(ValueError):
        train.validate(train.parser().parse_args(extra))


@pytest.mark.parametrize("kind", ["unilora", "unidora"])
def test_uni_is_not_exposed(kind):
    with pytest.raises(SystemExit):
        train.parser().parse_args(["--adapter-type", kind])


def test_seeded_candidate_epochs_use_equal_data_and_restore_rng():
    loader = DataLoader(dataset(9), batch_size=3, shuffle=True,
                        generator=torch.Generator().manual_seed(19))
    generator_before = loader.generator.get_state().clone()
    rng_before = torch.random.get_rng_state().clone()
    batches = []
    for _ in range(2):
        with search.seeded_stream(18, loader, "cpu"):
            batches.append(torch.cat([x for x, y in loader]))
            torch.rand(20)
    assert torch.equal(batches[0], batches[1])
    assert torch.equal(generator_before, loader.generator.get_state())
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_early_stopping_defers_test_until_explicit_evaluation(tmp_path, monkeypatch, tiny_workflow):
    import train_sfp_lora as old
    monkeypatch.setattr(train, "get_dataloaders", old.get_dataloaders)
    monkeypatch.setattr(train, "plot_training_curves", lambda *a, **k: {})
    monkeypatch.setattr(train, "plot_lr_schedule", lambda *a, **k: "omitted")
    monkeypatch.setattr(train, "plot_param_breakdown", lambda *a, **k: "omitted")
    evaluated = []
    def evaluation(model, loader, device, criterion):
        evaluated.append(loader)
        return 1.0, 50.0
    monkeypatch.setattr(train, "evaluate_full", evaluation)
    args = arguments(("lora", "direct", "default"), tmp_path)
    args.epochs = -1; args.max_epochs = 4; args.patience = 1; args.defer_test = True
    path = train.train(args)
    result = json.loads(path.read_text())
    assert len(evaluated) == 2  # Validation only; SNIP uses its own training calibration.
    assert result["epochs_trained"] == 2 and result["best_epoch"] == 1
    assert result["early_stopping_enabled"]
    assert result["final_test_acc"] is None and not result["test_evaluated"]
    result = train.evaluate_summary(path, "cpu")
    assert len(evaluated) == 3 and result["final_test_acc"] == 50.0
    assert result["test_evaluated"]
