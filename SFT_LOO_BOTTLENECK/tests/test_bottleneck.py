import copy
import json
from pathlib import Path
import sys
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_bottleneck as train
import loo_selection as selection
from bottleneck_filter import BottleneckFilterBlock


def test_reconstruction_improves_or_keeps_holdout_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(18)
    x = torch.randn(64, 4, 8)
    y = x + 0.2 * torch.nn.functional.gelu(x)
    block = BottleneckFilterBlock(8, 3, search_tokens=32, init_steps=5)
    block.init_from_pinv(x, y)
    assert block.fit_report['best_holdout_mse'] <= block.fit_report['initial_holdout_mse']
    assert block.fit_report['fit_images'] == 48
    assert block.fit_report['holdout_images'] == 16
    block(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert sum(p.numel() for p in block.parameters()) == 2*8*3 + 3 + 3*8


def dataset(n=64):
    generator = torch.Generator().manual_seed(42)
    return TensorDataset(torch.randn(n, 3, 32, 32, generator=generator), torch.arange(n) % 3)


CASES = [('lora', 'direct', 0), ('dora', 'direct', 0),
         ('paca', 'direct', 0), ('rpaca', 'direct', 0),
         ('paca', 'lora', 0), ('paca', 'dora', 0),
         ('rpaca', 'lora', 0), ('rpaca', 'dora', 0), ('none', 'direct', 0)]


@pytest.mark.parametrize('adapter,tuner,count', CASES)
def test_training_loo_peft_scope_exact_terms_and_saved_evaluation(adapter, tuner, count, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(18)
    original = VisionTransformer(img_size=32, patch_size=16, embed_dim=8,
                                 depth=3, num_heads=2, num_classes=3)
    saved = copy.deepcopy(original.state_dict())
    monkeypatch.setattr(train.timm, 'create_model', lambda *a, **k: copy.deepcopy(original))
    def loaders(args):
        return (DataLoader(dataset(), batch_size=32, shuffle=True,
                           generator=torch.Generator().manual_seed(args.seed)),
                DataLoader(dataset(8), batch_size=8), DataLoader(dataset(8), batch_size=8),
                3, ['a', 'b', 'c'])
    monkeypatch.setattr(train, 'get_dataloaders', loaders)
    monkeypatch.setattr(train, 'plot_training_curves', lambda *a, **k: {})
    monkeypatch.setattr(train, 'plot_lr_schedule', lambda *a, **k: 'omitted')
    monkeypatch.setattr(train, 'plot_param_breakdown', lambda *a, **k: 'omitted')
    args = train.parser().parse_args([
        '--device', 'cpu', '--epochs', '2', '--head-warmup-epochs', '1',
        '--output-dir', str(tmp_path), '--bottleneck-init-steps', '2',
        '--filter-rank', '2', '--init-tokens', '32',
        '--loo-samples', '17', '--loo-batch-size', '8',
        '--adapter-type', adapter, '--paca-tuner', tuner,
        '--lora-rank', '2', '--lora-alpha', '4', '--paca-rank', '3', '--paca-adapter-rank', '2'])
    path = train.train(args)
    summary = json.loads(path.read_text())
    assert summary['method'] == 'loo_bottleneck'
    assert summary['bottleneck_rank'] == 2
    assert summary['test_evaluated'] and summary['final_test_acc'] is not None
    replaced, sensitive = summary['replaced_block_idx'], summary['peft_block_idx']
    expected_replaced, expected_sensitive, _ = selection.choose_blocks(summary['loo_selection']['rows'])
    assert replaced == expected_replaced
    assert sensitive == (None if adapter == 'none' else expected_sensitive)
    model = train.restore_model(summary, 'cpu')
    selection.assert_trainable_scope(model, replaced, sensitive)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            # Wrappers rename original frozen weights; direct names cover untouched blocks/stem.
            if name in saved:
                assert torch.equal(p, saved[name])
    for filename in ['initial_filter_equations.json', 'filter_equations.json']:
        equation = json.loads((path.parent / filename).read_text())
        assert equation['type'] == 'residual_gelu_bottleneck'
    tested = train.evaluate_summary(path, 'cpu')
    assert tested['final_test_acc'] == summary['final_test_acc']
