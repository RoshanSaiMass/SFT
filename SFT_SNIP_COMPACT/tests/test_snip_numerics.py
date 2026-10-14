import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from timm.models.vision_transformer import VisionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snip_search as search
from single_filter_lora import SingleFilterBlock, freeze_non_trainable


def test_precision_aware_dense_fit_does_not_invert_fp32_noise():
    torch.set_num_threads(2)
    torch.manual_seed(18)
    n, width = 256, 16
    u, _ = torch.linalg.qr(torch.randn(n, width, dtype=torch.double))
    v, _ = torch.linalg.qr(torch.randn(width, width, dtype=torch.double))
    singular = torch.logspace(0, -7, width, dtype=torch.double)
    a = (u @ torch.diag(singular) @ v.T).float()
    b = a + 0.05 * torch.tanh(a @ torch.randn(width, width))
    expected = torch.linalg.pinv(a) @ b
    unsafe = torch.linalg.pinv(a.double()) @ b.double()
    block = SingleFilterBlock(width)
    search.initialize_dense_filter(block, a.reshape(64, 4, width), b.reshape(64, 4, width))
    assert block.weight.norm() < 10
    assert unsafe.norm() > 100 * block.weight.norm()
    assert torch.allclose(block.weight.T, expected, rtol=2e-3, atol=2e-3)
    assert block.fit_report['pinv_rtol'] == n * torch.finfo(torch.float32).eps
    assert block.fit_report['effective_rank'] < width
    assert block.fit_report['cls_mse'] is not None
    assert block.fit_report['patch_mse'] is not None
    assert block.fit_report['initialization_mse'] < 1e-7
    assert not block.fit_report['cutoff_is_paper_specified']


def test_float64_inputs_keep_float64_cutoff_and_exact_full_rank_fit():
    torch.manual_seed(3)
    x = torch.randn(64, 3, 8, dtype=torch.double)
    target = x @ torch.randn(8, 8, dtype=torch.double)
    block = SingleFilterBlock(8).double()
    search.initialize_dense_filter(block, x, target)
    assert torch.allclose(block(x), target, atol=1e-10)
    assert block.fit_report['pinv_rtol'] == 192 * torch.finfo(torch.float64).eps
    assert block.fit_report['effective_rank'] == 8


def test_zero_rank_initialization_fails_instead_of_silently_using_zero_filter():
    with pytest.raises(ValueError, match='No feature directions'):
        search.initialize_dense_filter(SingleFilterBlock(4), torch.zeros(64, 2, 4), torch.ones(64, 2, 4))


def test_scoring_differentiates_the_same_smoothed_loss_and_reports_components():
    torch.manual_seed(5)
    model = VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                             depth=3, num_heads=2, num_classes=3).eval()
    model.blocks[1] = SingleFilterBlock(8)
    freeze_non_trainable(model, [1])
    x, y = torch.randn(7, 3, 32, 32), torch.arange(7) % 3
    named = list(model.named_parameters())
    flags = [p.requires_grad for _, p in named]
    for _, p in named:
        p.requires_grad_(True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
    gradients = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    expected = sum(float((p.detach() * g).abs().double().sum())
                   for (_, p), g in zip(named, gradients) if g is not None)
    for (_, p), flag in zip(named, flags):
        p.requires_grad_(flag)
    result = search.snip_scores(model, DataLoader(TensorDataset(x, y), batch_size=3),
                                'cpu', label_smoothing=0.1, replaced=1)
    assert result['network_score'] == pytest.approx(expected, rel=1e-5)
    assert result['scoring_loss'] == pytest.approx(float(loss.detach()), rel=1e-5)
    assert result['network_score'] == pytest.approx(sum(result['component_scores'].values()))
    assert result['network_score'] == pytest.approx(result['trainable_snip'] + result['frozen_snip'])
    assert result['component_scores']['filter'] == pytest.approx(result['block_scores'][1])
    assert result['frozen_snip'] > 0
    assert [p.requires_grad for _, p in named] == flags
