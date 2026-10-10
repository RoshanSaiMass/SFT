import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import block_subtraction as blocks
import train_subtraction as trainer
from test_peft_integration import CASES, ADAPTERS
from select_removals import choose, collect
import run_removal_sweep as sweep


@pytest.fixture
def tiny(monkeypatch):
    torch.set_num_threads(2)
    models, test_calls = [], []
    def factory(*args, **kwargs):
        model = VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                                  depth=3, num_heads=2, num_classes=kwargs.get("num_classes", 3))
        models.append(model); return model
    def loaders(args):
        generator = torch.Generator().manual_seed(5)
        def loader(n):
            dataset = TensorDataset(torch.randn(n, 3, 32, 32, generator=generator),
                                    torch.randint(0, 3, (n,), generator=generator))
            return DataLoader(dataset, batch_size=16)
        return loader(64), loader(8), loader(8), 3, ["a", "b", "c"]
    monkeypatch.setattr(trainer.timm, "create_model", factory)
    monkeypatch.setattr(trainer, "get_dataloaders", loaders)
    monkeypatch.setattr(trainer, "plot_training_curves", lambda *a, **k: {})
    monkeypatch.setattr(trainer, "plot_lr_schedule", lambda *a, **k: "synthetic_plot")
    monkeypatch.setattr(trainer, "plot_param_breakdown", lambda *a, **k: "synthetic_plot")
    return models


def options(tmp_path, case=("lora", "direct", "default"), method="drop-one"):
    kind, tuner, init = case
    args = trainer.parser().parse_args([
        "--method", method, "--remove-block", "1" if method == "drop-one" else "-1",
        "--adapter-type", kind, "--paca-tuner", tuner, "--init-method", init,
        "--lora-rank", "2", "--paca-rank", "3", "--paca-adapter-rank", "2",
        "--unilora-dim", "11", "--lora-alpha", "4", "--device", "cpu", "--epochs", "2",
        "--output-dir", str(tmp_path), "--min-correlation", "-1", "--distill-epochs", "2"])
    return args


def test_manifest_covers_every_method_seed_and_block(tmp_path):
    args = options(tmp_path)
    args.adapter_types = trainer.ADAPTERS
    args.seeds = [18, 19]
    args.blocks = list(range(12))
    manifest = json.loads(sweep.build_manifest(args).read_text())
    assert len(manifest["tasks"]) == 144
    combinations = {(task["config"]["adapter_type"], task["config"]["seed"],
                     task["config"]["remove_block"]) for task in manifest["tasks"]}
    assert len(combinations) == 144
    assert all(task["config"]["defer_test"] for task in manifest["tasks"])
    assert len({task["task_dir"] for task in manifest["tasks"]}) == 144


def test_manifest_worker_resume_and_failure_are_recorded(tiny, tmp_path, monkeypatch):
    args = options(tmp_path)
    args.adapter_types = ["lora"]
    args.seeds = [18]
    args.blocks = [1]
    task = json.loads(sweep.build_manifest(args).read_text())["tasks"][0]
    assert sweep.execute(task)
    monkeypatch.setattr(sweep.trainer, "train", lambda *a: pytest.fail("Completed task retrained"))
    assert sweep.execute(task, resume=True)
    status = Path(task["task_dir"]) / "task_status.json"
    assert json.loads(status.read_text())["resumed"]
    monkeypatch.setattr(sweep.trainer, "train", lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic failure")))
    assert not sweep.execute(task)
    assert json.loads(status.read_text())["state"] == "failed"


def test_constant_outputs_cannot_select_a_pair():
    model = FeatureModel()
    loader = DataLoader(TensorDataset(torch.ones(4, 3, 4), torch.zeros(4)), batch_size=2)
    rows = blocks.adjacent_correlations(model, loader, "cpu")
    assert rows[0]["pearson"] is None
    with pytest.raises(ValueError, match="No requested"):
        blocks.select_pair(rows)


@pytest.mark.parametrize("policy", ["target", "none"])
def test_layernorm_training_scope(tiny, tmp_path, policy):
    model = trainer.timm.create_model("tiny", num_classes=3)
    args = options(tmp_path, method="correlation")
    args.train_layernorms = policy
    trainer.make_student(model, args, removed=2, retained=1)
    assert not model.blocks[0].norm1.weight.requires_grad
    assert model.blocks[1].norm1.weight.requires_grad == (policy == "target")
    assert model.norm.weight.requires_grad == (policy == "target")
    assert model.head.weight.requires_grad


@pytest.mark.parametrize("index", [0, 1, 2])
def test_deletion_is_physical_and_original_index_map_is_correct(index):
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=8, depth=3, num_heads=2, num_classes=3)
    originals = list(model.blocks)
    record = blocks.delete_block(model, index)
    assert len(model.blocks) == 2
    assert record["remaining_original_indices"] == [i for i in range(3) if i != index]
    assert all(block is originals[i] for block, i in zip(model.blocks, record["remaining_original_indices"]))
    assert all(block is not originals[index] for block in model.blocks)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 3)


class Scale(nn.Module):
    def forward(self, x):
        return 2 * x + 1


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.Sequential(nn.Identity(), Scale())
    def forward(self, x):
        return self.blocks(x)


def test_correlation_is_pearson_not_equivalence_and_hooks_are_cleaned():
    model = FeatureModel().train()
    loader = DataLoader(TensorDataset(torch.randn(7, 3, 4), torch.zeros(7)), batch_size=3)
    row = blocks.adjacent_correlations(model, loader, "cpu")[0]
    assert row["pearson"] == pytest.approx(1, abs=1e-6)
    assert row["rmse"] > 0.1
    assert row["samples"] == row["valid_samples"] == 7
    assert model.training
    assert not any(module._forward_hooks for module in model.modules())
    assert blocks.select_pair([row])["removed_block"] == 1


def test_no_eligible_pair_is_not_silently_removed():
    with pytest.raises(ValueError, match="No requested adjacent pair"):
        blocks.select_pair([dict(retained_block=0, removed_block=1, pearson=0.8)], 0.9)


def test_cached_calibration_does_not_advance_rng_or_loader_generator():
    generator = torch.Generator().manual_seed(8)
    loader = DataLoader(TensorDataset(torch.arange(60).reshape(20, 3), torch.arange(20)),
                        batch_size=4, shuffle=True, generator=generator)
    before = generator.get_state().clone(); torch_before = torch.random.get_rng_state().clone()
    py_before, np_before = random.getstate(), np.random.get_state()
    cached = blocks.calibration_images(loader, 7)
    assert len(cached) == 7
    assert torch.equal(before, generator.get_state())
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert py_before == random.getstate()
    assert np.array_equal(np_before[1], np.random.get_state()[1])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", ["correlation", "drop-one"])
def test_real_training_and_checkpoint_reload_all_peft_variants(tmp_path, tiny, case, method):
    args = options(tmp_path, case, method)
    path = trainer.train(args)
    summary = json.loads(path.read_text())
    model = tiny[0]
    assert len(model.blocks) == 2
    assert summary["param_breakdown"]["filter_block"] == 0
    assert summary["param_breakdown"]["trainable_backbone"] == 0
    assert summary["epochs_trained"] == 2
    assert summary["test_evaluated"]
    if method == "correlation":
        assert summary["removed_block_idx"] == summary["retained_block_idx"] + 1
        assert len(summary["structure"]["adapter_current_indices"]) == 1
        assert summary["distillation_epochs_applied"] == 2
        assert (path.parent / "adjacent_correlations.csv").exists()
    else:
        assert summary["removed_block_idx"] == 1
        assert len(summary["structure"]["adapter_current_indices"]) == 2
        assert summary["distillation_epochs_applied"] == 0
    wrappers = [module for module in model.modules() if isinstance(module, ADAPTERS)]
    assert len(wrappers) == 4 * len(summary["structure"]["adapter_current_indices"])
    x = torch.randn(2, 3, 32, 32)
    model.eval(); expected = model(x).detach()
    restored = trainer.restore_model(summary, "cpu")
    assert torch.allclose(restored(x), expected, atol=1e-6)


def test_distillation_reduces_feature_error_with_frozen_base_weights(tmp_path):
    class ToyBlock(nn.Module):
        def __init__(self):
            super().__init__(); self.qkv = nn.Linear(8, 8)
        def forward(self, x):
            return self.qkv(x)
    model = nn.Module(); model.blocks = nn.Sequential(ToyBlock(), ToyBlock())
    args = options(tmp_path, method="correlation")
    args.distill_epochs = 12; args.distill_lr = 0.02
    x = torch.randn(64, 3, 8)
    y = model.blocks(x).detach()
    trainer.make_student(model, args, 1, 0)
    block = model.blocks[0]
    frozen = block.qkv.base_layer.weight.detach().clone()
    before = float((block(x) - y).square().mean().detach())
    blocks.distill_pair(block, TensorDataset(x, y), args, "cpu")
    after = float((block(x) - y).square().mean().detach())
    assert after < before
    assert torch.equal(frozen, block.qkv.base_layer.weight)


def test_validation_selection_ignores_higher_test_scores_and_uses_seed_mean():
    def summary(block, seed, val, test):
        return dict(method="drop-one", dataset="pets", adapter_type="lora", seed=seed,
                    removed_block_idx=block, best_val_acc=val, final_test_acc=test,
                    config=dict(seed=seed, remove_block=block, adapter_type="lora"),
                    structure=dict(remaining_original_indices=[i for i in range(3) if i != block]))
    rows = [summary(b, s, val, test) for b, values in enumerate([(80, 99), (90, 70), (85, 95)])
            for s, val in [(18, values[0] - 1), (42, values[0] + 1)] for test in [values[1]]]
    winner = choose(rows, seeds=[18, 42])[0]
    assert winner["removed_block_idx"] == 1
    assert winner["mean_best_val_acc"] == 90
    with pytest.raises(ValueError, match="Incomplete removal sweep"):
        choose(rows[:-1], seeds=[18, 42])


def test_deferred_test_and_selected_checkpoint_evaluation(tmp_path, tiny, monkeypatch):
    paths = []
    for index in range(3):
        args = options(tmp_path)
        args.remove_block = index; args.defer_test = True
        path = trainer.train(args); paths.append(path)
        assert json.loads(path.read_text())["final_test_acc"] is None
    selected = collect(tmp_path, seeds=[18], blocks=[0, 1, 2], evaluate_selected=True, device="cpu")
    winner = selected[0]["removed_block_idx"]
    for index, path in enumerate(paths):
        summary = json.loads(path.read_text())
        assert summary["test_evaluated"] == (index == winner)
    assert (tmp_path / "compiled_results/results_all.csv").exists()
    assert (tmp_path / "compiled_results/best_removals.csv").exists()


def test_collector_detects_entire_missing_method_from_manifest(tmp_path, tiny):
    args = options(tmp_path)
    args.adapter_types = ["lora", "dora"]
    args.seeds = [18]
    args.blocks = [1]
    manifest = json.loads(sweep.build_manifest(args).read_text())
    assert sweep.execute(manifest["tasks"][0])
    with pytest.raises(ValueError, match="entire planned configurations missing"):
        collect(manifest["root"], evaluate_selected=True, device="cpu")
    assert (Path(manifest["root"]) / "compiled_results/results_all.csv").exists()


@pytest.mark.parametrize("flags", [
    ["--method", "drop-one"], ["--method", "correlation", "--remove-block", "1"],
    ["--lora-rank", "0"], ["--min-correlation", "nan"],
    ["--adapter-type", "paca", "--init-method", "loftq"],
    ["--adapter-type", "unilora", "--lora-ortho-lambda1", "1"],
    ["--adapter-type", "paca", "--paca-tuner", "dora"],
])
def test_invalid_configs_rejected(flags):
    with pytest.raises(ValueError):
        trainer.validate(trainer.parser().parse_args(flags))
