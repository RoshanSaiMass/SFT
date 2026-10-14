import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compressed_filters import OPERATORS, SymbolicFilterBlock, basis
from train_loo_compact import parser, validate


@pytest.mark.parametrize('name, function', [('gelu', F.gelu), ('silu', F.silu), ('softsign', F.softsign)])
def test_new_basis_functions_have_finite_values_and_gradients(name, function):
    x = torch.tensor([-100., -3., 0., 3., 100.], dtype=torch.double, requires_grad=True)
    result = basis(x, name)
    torch.testing.assert_close(result, function(x))
    result.sum().backward()
    assert torch.isfinite(result).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('name', ['linear', 'square', 'cube', 'tanh', 'sin', 'constant'])
def test_existing_checkpoint_operator_ids_still_decode_the_same_function(name):
    old_names = ('linear', 'square', 'cube', 'tanh', 'sin', 'constant')
    assert OPERATORS[:6] == old_names
    block = SymbolicFilterBlock(8, 2)
    with torch.no_grad():
        block.up.weight.normal_(std=0.1)
        block.coeff[0].fill_(0.7)
        block.operator_ids[0] = old_names.index(name)
    saved = block.state_dict()
    restored = SymbolicFilterBlock(8, 2)
    restored.load_state_dict(saved)
    x = torch.randn(4, 3, 8)
    z = restored.down(x) / restored.scale
    expected = x + restored.up(restored.scale * 0.7 * basis(z, name))
    assert torch.equal(restored(x), expected)


@pytest.mark.parametrize('limit, candidates', [(3, 129), (5, 381), (7, 501), (9, 511)])
def test_loo_accepts_and_searches_each_expanded_term_limit(limit, candidates):
    torch.set_num_threads(2)
    args = parser().parse_args(['--symbolic-max-terms', str(limit), '--filter-rank', '128', '--lora-rank', '8'])
    validate(args)
    torch.manual_seed(18)
    x = torch.randn(64, 3, 2)
    y = x + 0.3 * F.gelu(x)
    block = SymbolicFilterBlock(2, 1, max_terms=limit, search_tokens=64)
    block.init_from_pinv(x, y)
    assert len(block.fit_report['candidates']) == candidates
    assert sum(p.numel() for p in block.parameters()) == 2 * 2 + 2 + limit
    assert block.fit_report['operator_library'] == list(OPERATORS)
    block(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    restored = SymbolicFilterBlock(2, 1, max_terms=limit)
    restored.load_state_dict(block.state_dict())
    assert torch.equal(block(x), restored(x))


@pytest.mark.parametrize('limit', [0, 10])
def test_term_limit_outside_library_is_rejected(limit):
    with pytest.raises(ValueError):
        validate(parser().parse_args(['--symbolic-max-terms', str(limit)]))
