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
import train_loo_all_peft as train
import loo_selection as selection
import single_filter_lora as sf


def tiny(depth=12):
    torch.set_num_threads(2)
    return VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                             depth=depth, num_heads=2, num_classes=3)


def data(n=64):
    generator = torch.Generator().manual_seed(42)
    return TensorDataset(torch.randn(n, 3, 32, 32, generator=generator), torch.arange(n) % 3)


def arguments(path, adapter="lora"):
    return train.parser().parse_args([
        "--device", "cpu", "--epochs", "2", "--output-dir", str(path),
        "--filter-rank", "2", "--symbolic-search-tokens", "32",
        "--adapter-type", adapter, "--head-warmup-epochs", "1",
        "--loo-samples", "17", "--loo-batch-size", "7",
    ])


def patch_workflow(monkeypatch):
    original = tiny()
    saved = copy.deepcopy(original.state_dict())
    monkeypatch.setattr(train.timm, "create_model", lambda *a, **k: copy.deepcopy(original))
    monkeypatch.setattr(train, "get_dataloaders", lambda args: (
        DataLoader(data(), batch_size=23, shuffle=True, generator=torch.Generator().manual_seed(args.seed)),
        DataLoader(data(8), batch_size=8), DataLoader(data(8), batch_size=8), 3, ["a", "b", "c"]))
    monkeypatch.setattr(train, "plot_training_curves", lambda *a, **k: {})
    monkeypatch.setattr(train, "plot_lr_schedule", lambda *a, **k: "omitted")
    monkeypatch.setattr(train, "plot_param_breakdown", lambda *a, **k: "omitted")
    return saved


@pytest.mark.parametrize("adapter", ["lora", "dora", "paca", "rpaca", "none"])
def test_twelve_block_training_scope_metadata_and_reload(tmp_path, monkeypatch, adapter):
    saved = patch_workflow(monkeypatch)
    args = arguments(tmp_path, adapter)
    if adapter in ("paca", "rpaca"):
        args.paca_rank = 4
    path = train.train(args)
    result = json.loads(path.read_text())
    replaced = result["replaced_block_idx"]
    expected_targets = [i for i in range(12) if i != replaced] if adapter != "none" else []
    assert result["peft_block_indices"] == expected_targets
    assert result["peft_block_idx"] is None
    assert result["structure"]["adapter_original_indices"] == expected_targets
    assert result["loo_selection"]["peft_blocks"] == expected_targets
    assert [r["block"] for r in result["loo_selection"]["rows"] if r["peft_target"]] == expected_targets
    assert not result["loo_selection"]["used_validation"] and not result["loo_selection"]["used_test"]
    assert result["loo_selection"]["screening_forward_images"] == 13 * 17
    rows = result["loo_selection"]["rows"]
    assert replaced == min(rows, key=lambda r: (r["loss_increase"], r["block"]))["block"]
    assert result["epochs_trained"] == 2 and result["test_evaluated"]
    assert result["method"] == "loo-all-peft"
    assert result["param_breakdown"]["trainable_backbone"] == 0
    model = train.restore_model(result, "cpu")
    assert len(model.blocks) == 12
    assert not sf.adapter_parameter_ids(model.blocks[replaced])
    for index in range(12):
        if index != replaced:
            assert bool(sf.adapter_parameter_ids(model.blocks[index])) == (adapter != "none")
            if adapter in ("lora", "dora"):
                wrappers = [m for m in model.blocks[index].modules() if isinstance(m, sf.LoRALinear)]
                assert len(wrappers) == 4 and all(m.rank == 4 for m in wrappers)
                for layer_name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                    wrapper = model.blocks[index].get_submodule(layer_name)
                    assert torch.equal(wrapper.base_layer.weight, saved[f"blocks.{index}.{layer_name}.weight"])
    allowed = sf.adapter_parameter_ids(model)
    allowed.update(id(p) for p in model.blocks[replaced].parameters())
    allowed.update(id(p) for module in model.modules() if isinstance(module, nn.LayerNorm) for p in module.parameters())
    allowed.update(id(p) for p in model.head.parameters())
    assert {id(p) for p in model.parameters() if p.requires_grad} == allowed
    selection.assert_trainable_scope(model, replaced, expected_targets)
    again = train.restore_model(result, "cpu")
    assert torch.allclose(model(data(2).tensors[0]), again(data(2).tensors[0]), atol=1e-6)
    assert train.evaluate_summary(path, "cpu")["final_test_acc"] == result["final_test_acc"]


def test_scope_rejects_extra_weight_and_missing_adapter(tmp_path):
    model = tiny()
    args = arguments(tmp_path)
    selection.install_compact(model, args, 0)
    structure = train.apply_peft(model, args, 0)
    selection.assert_trainable_scope(model, 0, structure["adapter_original_indices"])
    model.blocks[1].attn.qkv.base_layer.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Unexpected trainable"):
        selection.assert_trainable_scope(model, 0, structure["adapter_original_indices"])
    model.blocks[1].attn.qkv.base_layer.weight.requires_grad_(False)
    for name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
        parent, attribute = name.rsplit(".", 1)
        wrapper = model.blocks[1].get_submodule(name)
        setattr(model.blocks[1].get_submodule(parent), attribute, wrapper.base_layer)
    with pytest.raises(RuntimeError, match="Missing adapters"):
        selection.assert_trainable_scope(model, 0, structure["adapter_original_indices"])


def test_early_stopping_and_deferred_test(tmp_path, monkeypatch):
    patch_workflow(monkeypatch)
    calls = []
    def evaluate(model, loader, device, criterion):
        calls.append(loader)
        return 1.0, 50.0
    monkeypatch.setattr(train, "evaluate_full", evaluate)
    args = arguments(tmp_path)
    args.epochs = -1; args.max_epochs = 4; args.patience = 1; args.defer_test = True
    path = train.train(args)
    result = json.loads(path.read_text())
    assert result["epochs_trained"] == 2 and result["best_epoch"] == 1
    assert not result["test_evaluated"] and result["final_test_acc"] is None
    assert len(calls) == 2
    result = train.evaluate_summary(path, "cpu")
    assert result["test_evaluated"] and len(calls) == 3


def test_pets_parameter_budget_and_every_original_block():
    import timm
    model = timm.create_model(train.BACKBONE, pretrained=False, num_classes=37)
    original_count = sum(p.numel() for p in model.parameters())
    args = train.parser().parse_args(["--device", "cpu"])
    assert args.filter_rank == 16 and args.lora_rank == 4 and args.lora_alpha == 8
    selection.install_compact(model, args, 11)
    structure = train.apply_peft(model, args, 11)
    assert structure["adapter_original_indices"] == list(range(11))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert original_count == 85_827_109 and count == 629_829
    assert 100 * count / original_count < 0.8
    selection.assert_trainable_scope(model, 11, structure["adapter_original_indices"])
