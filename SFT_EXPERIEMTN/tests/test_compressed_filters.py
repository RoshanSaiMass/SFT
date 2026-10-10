import copy
import io
import sys
from pathlib import Path

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compressed_filters import LowRankFilterBlock, SymbolicFilterBlock, replace_filter
from single_filter_lora import count_parameter_breakdown, freeze_non_trainable


@pytest.mark.parametrize("width,rank", [(8, 2), (768, 8), (768, 16), (768, 32)])
@pytest.mark.parametrize("kind", [LowRankFilterBlock, SymbolicFilterBlock])
def test_parameters_include_projections_and_symbolic_coefficients(width, rank, kind):
    block = kind(width, rank)
    expected = 2 * width * rank + width + (2 * rank if kind is SymbolicFilterBlock else 0)
    assert sum(p.numel() for p in block.parameters()) == expected
    assert all(p.shape != (width, width) for p in block.parameters())
    assert all(b.shape != (width, width) for b in block.buffers())


def test_full_rank_pinv_matches_dense_and_low_rank_truncation():
    torch.manual_seed(7)
    x = torch.randn(64, 3, 8, dtype=torch.double)
    target = torch.randn(8, 8, dtype=torch.double)
    y = x @ target
    full = LowRankFilterBlock(8, 8).double(); full.init_from_pinv(x, y)
    assert torch.allclose(full(x), y, atol=1e-10)
    low = LowRankFilterBlock(8, 2).double(); low.init_from_pinv(x, y)
    u, s, vh = torch.linalg.svd(target)
    assert torch.allclose(low(x), x @ ((u[:, :2] * s[:2]) @ vh[:2]), atol=1e-10)


def test_symbolic_search_recovers_nonlinear_expression():
    torch.manual_seed(9)
    x = torch.rand(64, 5, 1, dtype=torch.double) * 2 - 1
    y = x + 0.8 * x + 0.35 * x.square()
    block = SymbolicFilterBlock(1, 1, penalty=1e-6).double()
    block.init_from_pinv(x, y)
    assert set(block.fit_report["selected_terms"]) == {"linear", "square"}
    assert torch.allclose(block(x), y, atol=1e-9)
    assert block.fit_report["fit_images"] == 48
    assert block.fit_report["heldout_images"] == 16


@pytest.mark.parametrize("kind", [LowRankFilterBlock, SymbolicFilterBlock])
def test_gradient_updates_and_checkpoint_roundtrip(kind):
    torch.manual_seed(11)
    x = torch.randn(64, 3, 8)
    y = x + 0.4 * x.tanh()
    block = kind(8, 2); block.init_from_pinv(x, y)
    optimizer = torch.optim.AdamW(block.parameters(), lr=0.01)
    before = copy.deepcopy(block.state_dict())
    loss = (block(x) - y).square().mean(); loss.backward(); optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert any(not torch.equal(v, block.state_dict()[k]) for k, v in before.items())
    buf = io.BytesIO(); torch.save(block.state_dict(), buf); buf.seek(0)
    restored = kind(8, 2); restored.load_state_dict(torch.load(buf, weights_only=True))
    assert torch.equal(block(x), restored(x))


@pytest.mark.parametrize("filter_type", ["lowrank", "symbolic"])
def test_multi_replacement_keeps_dtype_device_and_freeze_policy(filter_type):
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=8, depth=3,
                              num_heads=2, num_classes=3).double()
    for index in (0, 2):
        block = replace_filter(model, index, filter_type=filter_type, rank=2)
        assert block.down.weight.dtype == torch.double
        block.init_from_pinv(torch.randn(64, 5, 8, dtype=torch.double),
                             torch.randn(64, 5, 8, dtype=torch.double))
        assert torch.isfinite(model(torch.randn(2, 3, 32, 32, dtype=torch.double))).all()
    freeze_non_trainable(model, [0, 2])
    assert all(p.requires_grad for i in (0, 2) for p in model.blocks[i].parameters())
    assert not model.blocks[1].attn.qkv.weight.requires_grad
    assert count_parameter_breakdown(model, [0, 2])["filter_block"] == sum(
        p.numel() for i in (0, 2) for p in model.blocks[i].parameters())


def test_all_token_local_filters_make_cls_predictions_image_independent():
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=8, depth=3,
                              num_heads=2, num_classes=3).eval()
    for index in range(3):
        replace_filter(model, index, filter_type="dense")
    predictions = model(torch.randn(4, 3, 32, 32))
    assert torch.allclose(predictions, predictions[:1].expand_as(predictions))


@pytest.mark.parametrize("filter_type", ["dense", "lowrank", "symbolic"])
def test_constructor_does_not_change_training_sampling_rng(filter_type):
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=8, depth=3,
                              num_heads=2, num_classes=3)
    before = torch.random.get_rng_state().clone()
    replace_filter(model, 1, filter_type=filter_type, rank=2)
    assert torch.equal(torch.random.get_rng_state(), before)
