import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loo_selection as selection
import train_loo_compact as train
import single_filter_lora as sf
from calibration import calibration_images
from test_peft_integration import CASES, ADAPTERS, expected_adapter_ids


def tiny():
    torch.set_num_threads(2)
    return VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                             depth=3, num_heads=2, num_classes=3)


def dataset(n=64):
    generator = torch.Generator().manual_seed(42)
    return TensorDataset(torch.randn(n, 3, 32, 32, generator=generator), torch.arange(n) % 3)


class LoopModel(nn.Module):
    """A forward loop supports ModuleList; timm ViT instead calls Sequential."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(3)])
        self.head = nn.Linear(3, 3)

    def forward(self, x):
        x = x.mean((2, 3))
        for block in self.blocks:
            x = block(x).tanh()
        return self.head(x)


def arguments(case, output, filter_type="symbolic"):
    kind, tuner, init = case
    return train.parser().parse_args([
        "--device", "cpu", "--epochs", "2", "--output-dir", str(output),
        "--filter-type", filter_type, "--filter-rank", "2", "--symbolic-search-tokens", "32",
        "--adapter-type", kind, "--paca-tuner", tuner, "--init-method", init,
        "--lora-rank", "2", "--lora-alpha", "4", "--paca-rank", "3",
        "--paca-adapter-rank", "2", "--loo-samples", "17", "--loo-batch-size", "7",
        "--head-warmup-epochs", "2"])


@pytest.mark.parametrize("container", [nn.Sequential, nn.ModuleList])
def test_each_removal_matches_independent_copy_and_preserves_original(container):
    model = tiny() if container is nn.Sequential else LoopModel()
    data = dataset(7)
    original = model.blocks
    model.train(); model.blocks[1].eval()
    modes = [m.training for m in model.modules()]
    weights = copy.deepcopy(model.state_dict())
    parameters = list(model.parameters())
    for index, p in enumerate(parameters):
        p.requires_grad_(index % 2 == 0); p.grad = torch.ones_like(p)
    flags = [p.requires_grad for p in parameters]
    result = selection.screen_removals(model, data, 3, "cpu", 0.1)
    assert model.blocks is original
    model_copy = copy.deepcopy(model); model_copy.eval()
    x, y = data.tensors
    baseline = float(F.cross_entropy(model_copy(x), y, label_smoothing=0.1).detach())
    assert result["baseline_loss"] == pytest.approx(baseline, rel=1e-6)
    for row in result["rows"]:
        candidate = copy.deepcopy(model); candidate.eval()
        remaining = [b for i, b in enumerate(candidate.blocks) if i != row["block"]]
        candidate.blocks = nn.Sequential(*remaining) if container is nn.Sequential else nn.ModuleList(remaining)
        expected = float(F.cross_entropy(candidate(x), y, label_smoothing=0.1).detach())
        assert row["loss"] == pytest.approx(expected, rel=1e-6)
        assert row["loss_increase"] == pytest.approx(expected-baseline, abs=1e-6)
        assert row["samples"] == 7
    assert result["screening_forward_images"] == 28
    assert [m.training for m in model.modules()] == modes
    assert [p.requires_grad for p in parameters] == flags
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in parameters)
    assert all(torch.equal(value, weights[name]) for name, value in model.state_dict().items())
    other = selection.screen_removals(model, data, 7, "cpu", 0.1)
    assert [r["loss_increase"] for r in result["rows"]] == pytest.approx(
        [r["loss_increase"] for r in other["rows"]], abs=1e-6)


def test_failure_restores_container_modes_weights(monkeypatch):
    model = tiny(); original = model.blocks; modes = [m.training for m in model.modules()]
    state = copy.deepcopy(model.state_dict()); forward = model.forward
    def failure(x):
        if len(model.blocks) != 3:
            raise RuntimeError("test candidate failure")
        return forward(x)
    monkeypatch.setattr(model, "forward", failure)
    with pytest.raises(RuntimeError, match="candidate failure"):
        selection.screen_removals(model, dataset(4), 2, "cpu")
    assert model.blocks is original
    assert [m.training for m in model.modules()] == modes
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())


def test_signed_loss_ranking_not_absolute_value_and_tie_collision():
    rows = [dict(block=i, loss_increase=value) for i, value in enumerate([-2, 0.1, 0.6])]
    assert selection.choose_blocks(rows) == (0, 2, 2)
    assert selection.choose_blocks([dict(block=i, loss_increase=1) for i in range(3)]) == (0, 1, 0)
    for bad in [rows[:1], [rows[1], rows[2]], [rows[0], rows[0]],
                [rows[0], dict(block=1, loss_increase=float("nan"))]]:
        with pytest.raises(ValueError):
            selection.choose_blocks(bad)


def test_balanced_sampling_reproducible_coverage_and_exhaustion():
    pool = TensorDataset(torch.arange(20).float().reshape(20, 1), torch.tensor([0]*15+[1]*3+[2]*2))
    a, report = selection.balanced_calibration(pool, 6, 18)
    b, again = selection.balanced_calibration(pool, 6, 18)
    assert report == again and torch.equal(a.tensors[0], b.tensors[0])
    assert report["scoring_class_counts"] == {0: 2, 1: 2, 2: 2}
    full, report = selection.balanced_calibration(pool, 100, 18)
    assert len(full) == 20 and len(set(report["selected_pool_indices"])) == 20


def test_head_warmup_only_changes_classifier(tmp_path):
    model = tiny(); original = copy.deepcopy(model.state_dict())
    model.train(); model.blocks[0].eval()
    modes = [m.training for m in model.modules()]; flags = [p.requires_grad for p in model.parameters()]
    args = arguments(("lora", "direct", "default"), tmp_path)
    report = selection.warmup_head(model, DataLoader(dataset(), batch_size=16), args)
    changed = {name for name, value in model.state_dict().items() if not torch.equal(value, original[name])}
    assert changed and all(name.startswith("head.") for name in changed)
    assert [m.training for m in model.modules()] == modes
    assert [p.requires_grad for p in model.parameters()] == flags
    assert report["epochs"] == 2 and not report["backbone_updated"]


def test_cached_pool_preserves_sampling_rng():
    loader = DataLoader(dataset(), batch_size=7, shuffle=True, generator=torch.Generator().manual_seed(18))
    state = loader.generator.get_state().clone(); global_state = torch.random.get_rng_state().clone()
    a = calibration_images(loader, 35); b = calibration_images(loader, 35)
    assert torch.equal(a.tensors[0], b.tensors[0])
    assert torch.equal(state, loader.generator.get_state())
    assert torch.equal(global_state, torch.random.get_rng_state())


def patch_workflow(monkeypatch):
    original = tiny(); saved = copy.deepcopy(original.state_dict()); models = []
    def factory(*a, **k):
        model = copy.deepcopy(original); models.append(model); return model
    def loaders(args):
        return (DataLoader(dataset(), batch_size=23, shuffle=True,
                           generator=torch.Generator().manual_seed(args.seed)),
                DataLoader(dataset(8), batch_size=8), DataLoader(dataset(8), batch_size=8),
                3, ["a", "b", "c"])
    monkeypatch.setattr(train.timm, "create_model", factory)
    monkeypatch.setattr(train, "get_dataloaders", loaders)
    for name in ("plot_lr_schedule", "plot_param_breakdown"):
        monkeypatch.setattr(train, name, lambda *a, **k: "omitted")
    monkeypatch.setattr(train, "plot_training_curves", lambda *a, **k: {})
    return saved, models


@pytest.mark.parametrize("case", [c for c in CASES if not c[0].startswith("uni")] + [("none", "direct", "default")])
@pytest.mark.parametrize("filter_type", ["symbolic", "lowrank", "dense"])
def test_full_training_adapter_scope_frozen_weights_and_reload(case, filter_type, tmp_path, monkeypatch):
    saved, models = patch_workflow(monkeypatch)
    args = arguments(case, tmp_path, filter_type)
    scoring_models = []
    scoring = selection.screen_removals
    def check_original(model, *a, **k):
        assert len(model.blocks) == 3
        assert not any(isinstance(m, ADAPTERS) for m in model.modules())
        assert all(torch.equal(value, saved[name]) for name, value in model.state_dict().items()
                   if not name.startswith("head."))
        scoring_models.append(model)
        return scoring(model, *a, **k)
    monkeypatch.setattr(selection, "screen_removals", check_original)
    path = train.train(args); result = json.loads(path.read_text())
    assert len(scoring_models) == 1
    replaced, sensitive = result["replaced_block_idx"], result["peft_block_idx"]
    metadata = result["loo_selection"]
    expected_replaced, expected_sensitive, _ = selection.choose_blocks(metadata["rows"])
    assert replaced == expected_replaced
    assert sensitive == (None if case[0] == "none" else expected_sensitive)
    assert metadata["screening_forward_images"] == 4*17
    assert metadata["head_warmup"]["epochs"] == 2
    assert not metadata["used_validation"] and not metadata["used_test"]
    assert result["test_evaluated"] and result["final_test_acc"] is not None
    assert result["epochs_trained"] == 2 and result["param_breakdown"]["trainable_backbone"] == 0
    model = train.restore_model(result, "cpu")
    assert len(model.blocks) == 3
    expected = expected_adapter_ids(model)
    expected.update(id(p) for p in model.blocks[replaced].parameters())
    expected.update(id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters())
    expected.update(id(p) for p in model.head.parameters())
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected
    selection.assert_trainable_scope(model, replaced, sensitive)
    wrappers = [m for m in model.modules() if isinstance(m, ADAPTERS)]
    assert len(wrappers) == (0 if case[0] == "none" else 4)
    if wrappers:
        assert all(any(m is w for m in model.blocks[sensitive].modules()) for w in wrappers)
    for i in set(range(3)) - {replaced, sensitive}:
        for name, p in model.blocks[i].named_parameters():
            if not p.requires_grad:
                assert torch.equal(p, saved[f"blocks.{i}.{name}"])
    x = dataset(2).tensors[0]
    assert torch.allclose(model(x), models[0].eval()(x), atol=1e-6)
    again = train.restore_model(result, "cpu")
    assert torch.allclose(model(x), again(x), atol=1e-6)
    evaluated = train.evaluate_summary(path, "cpu")
    assert evaluated["final_test_acc"] == result["final_test_acc"]
    assert (path.parent / "loo_selection.json").is_file()
    assert (path.parent / "loo_block_scores.csv").is_file()
    assert (path.parent / "filter_equations.json").is_file()
    with pytest.raises(FileExistsError):
        train.train(args)


def test_early_stopping_restores_best_and_defers_test(tmp_path, monkeypatch):
    patch_workflow(monkeypatch)
    calls = []; checkpoints = []
    def evaluate(model, loader, device, criterion):
        calls.append(loader); checkpoints.append(copy.deepcopy(model.state_dict()))
        return 1.0, 50.0
    monkeypatch.setattr(train, "evaluate_full", evaluate)
    args = arguments(("lora", "direct", "default"), tmp_path)
    args.epochs = -1; args.max_epochs = 4; args.patience = 1; args.defer_test = True
    path = train.train(args); result = json.loads(path.read_text())
    assert len(calls) == 2 and result["best_epoch"] == 1 and result["epochs_trained"] == 2
    assert result["final_test_acc"] is None and not result["test_evaluated"]
    best = torch.load(result["checkpoint_path"], weights_only=True)
    assert all(torch.equal(value, checkpoints[0][name]) for name, value in best.items())
    evaluated = train.evaluate_summary(path, "cpu")
    assert len(calls) == 3 and evaluated["final_test_acc"] == 50.0


def test_scope_guard_detects_dense_backbone_training(tmp_path):
    model = tiny(); args = arguments(("lora", "direct", "default"), tmp_path)
    selection.install_compact(model, args, 0); train.apply_top_peft(model, args, 0, 2)
    selection.assert_trainable_scope(model, 0, 2)
    model.blocks[1].mlp.fc1.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="scope"):
        selection.assert_trainable_scope(model, 0, 2)


@pytest.mark.parametrize("extra", [["--loo-samples", "0"], ["--loo-batch-size", "0"],
    ["--head-warmup-epochs", "-1"], ["--head-warmup-lr", "nan"], ["--epochs", "0"],
    ["--adapter-type", "paca", "--init-method", "loftq"],
    ["--adapter-type", "none", "--lora-ortho-lambda1", "1"]])
def test_invalid_settings(extra):
    with pytest.raises(ValueError):
        train.validate(train.parser().parse_args(extra))


@pytest.mark.parametrize("kind", ["unilora", "unidora"])
def test_uni_excluded(kind):
    with pytest.raises(SystemExit):
        train.parser().parse_args(["--adapter-type", kind])


def test_too_few_initialization_images_fail_before_warmup(tmp_path):
    args = arguments(("lora", "direct", "default"), tmp_path)
    with pytest.raises(ValueError, match="64 training"):
        selection.select_compact_candidate(tiny(), DataLoader(dataset(8)), dataset(8), args)


def test_zero_head_warmup_is_explicit_and_changes_no_weights(tmp_path):
    model = tiny(); before = copy.deepcopy(model.state_dict())
    args = arguments(("lora", "direct", "default"), tmp_path); args.head_warmup_epochs = 0
    report = selection.warmup_head(model, DataLoader(dataset()), args)
    assert report["epochs"] == 0 and not report["history"]
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_ranking_reproducible_across_peft_choices(tmp_path, monkeypatch):
    patch_workflow(monkeypatch)
    summaries = []
    for case in [("lora", "direct", "default"), ("paca", "dora", "default")]:
        args = arguments(case, tmp_path)
        summaries.append(json.loads(train.train(args).read_text()))
    a, b = [s["loo_selection"] for s in summaries]
    assert a["rows"] == b["rows"]
    assert a["head_warmup"]["history"] == b["head_warmup"]["history"]
    assert a["calibration"] == b["calibration"]


def test_screening_rejects_empty_data_and_nonfinite_loss(monkeypatch):
    model = tiny(); original = model.blocks
    with pytest.raises(ValueError, match="Empty"):
        selection.screen_removals(model, dataset(0), 3, "cpu")
    monkeypatch.setattr(model, "forward", lambda x: torch.full((len(x), 3), float("nan")))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        selection.screen_removals(model, dataset(4), 3, "cpu")
    assert model.blocks is original


def test_twelve_block_indices_and_peft_at_last_block(tmp_path):
    torch.set_num_threads(2)
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                              depth=12, num_heads=2, num_classes=3)
    original = model.blocks
    result = selection.screen_removals(model, dataset(7), 3, "cpu")
    assert model.blocks is original
    assert [r["block"] for r in result["rows"]] == list(range(12))
    assert result["screening_forward_images"] == 13*7
    assert all(r["loss_increase"] == pytest.approx(r["loss"]-result["baseline_loss"])
               for r in result["rows"])
    # Make the last original block the PEFT target to exercise indices >= 10.
    rows = [dict(block=i, loss_increase=float(i)) for i in range(12)]
    replaced, sensitive, _ = selection.choose_blocks(rows)
    assert (replaced, sensitive) == (0, 11)
    args = arguments(("dora", "direct", "default"), tmp_path)
    selection.install_compact(model, args, replaced)
    train.apply_top_peft(model, args, replaced, sensitive)
    selection.assert_trainable_scope(model, replaced, sensitive)
    assert len(model.blocks) == 12
    assert len([m for m in model.blocks[11].modules() if isinstance(m, ADAPTERS)]) == 4
