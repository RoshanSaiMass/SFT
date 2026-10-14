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
import compressed_filters as filters
import train_num_function as train
import loo_selection as selection


@pytest.mark.parametrize('count, combinations', [(3, 84), (5, 126), (7, 36), (9, 1)])
def test_exact_count_search_enumerates_only_requested_size_and_restores(count, combinations):
    torch.set_num_threads(2)
    torch.manual_seed(18)
    x = torch.randn(64, 3, 2)
    y = x + 0.3 * torch.nn.functional.gelu(x)
    block = filters.SymbolicFilterBlock(2, 1, max_terms=count, search_tokens=64)
    block.init_from_pinv(x, y)
    rows = block.fit_report['candidates']
    assert len(rows) == combinations
    assert all(len(row['terms']) == count for row in rows)
    assert len({tuple(row['terms']) for row in rows}) == combinations
    assert len(block.fit_report['selected_terms']) == count
    best = min(rows, key=lambda row: row['selection_objective'])
    assert block.fit_report['selected_terms'] == best['terms']
    if count == 9:
        assert block.fit_report['selected_terms'] == list(filters.OPERATORS)
    assert block.equation_report()['selection_mode'] == 'exact_count'
    assert sum(p.numel() for p in block.parameters()) == 6 + count
    clone = filters.SymbolicFilterBlock(2, 1, max_terms=count)
    clone.load_state_dict(block.state_dict())
    assert torch.equal(block(x), clone(x))
    block(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


@pytest.mark.parametrize('flag', ['--num-functions', '--symbolic-max-terms'])
def test_cli_count_and_compatibility_alias_have_exact_semantics(flag):
    args = train.parser().parse_args([flag, '7'])
    train.validate(args)
    assert args.symbolic_max_terms == 7


@pytest.mark.parametrize('count', [0, 10])
def test_invalid_count_rejected(count):
    with pytest.raises(ValueError):
        train.validate(train.parser().parse_args(['--num-functions', str(count)]))


def test_other_filter_architectures_are_not_silently_selected():
    with pytest.raises(SystemExit):
        train.parser().parse_args(['--filter-type', 'dense'])


def dataset(n=64):
    generator = torch.Generator().manual_seed(42)
    return TensorDataset(torch.randn(n, 3, 32, 32, generator=generator), torch.arange(n) % 3)


CASES = [('lora', 'direct', 3), ('lora', 'direct', 5),
         ('lora', 'direct', 7), ('lora', 'direct', 9),
         ('dora', 'direct', 3), ('dora', 'direct', 5),
         ('dora', 'direct', 7), ('dora', 'direct', 9),
         ('paca', 'direct', 3), ('rpaca', 'direct', 3),
         ('paca', 'lora', 3), ('paca', 'dora', 3),
         ('rpaca', 'lora', 3), ('rpaca', 'dora', 3), ('none', 'direct', 3)]


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
        '--output-dir', str(tmp_path), '--num-functions', str(count),
        '--filter-rank', '2', '--symbolic-search-tokens', '32',
        '--loo-samples', '17', '--loo-batch-size', '8',
        '--adapter-type', adapter, '--paca-tuner', tuner,
        '--lora-rank', '2', '--lora-alpha', '4', '--paca-rank', '3', '--paca-adapter-rank', '2'])
    path = train.train(args)
    summary = json.loads(path.read_text())
    assert summary['method'] == 'num_function'
    assert summary['num_functions'] == count
    assert len(summary['selected_terms']) == count
    assert summary['test_evaluated'] and summary['final_test_acc'] is not None
    assert summary['config']['function_selection'] == 'exact_count'
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
        assert equation['selected_term_count'] == equation['requested_terms'] == count
    tested = train.evaluate_summary(path, 'cpu')
    assert tested['final_test_acc'] == summary['final_test_acc']
