import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data
import single_filter_lora as filters
import analyse_results as analysis


def test_filter_pseudoinverse_handles_rank_deficiency_and_transpose():
    torch.manual_seed(7)
    x = torch.randn(64, 3, 4)
    x[..., 3] = x[..., 0]  # singular input, no normal-equation assumption
    y = x @ torch.randn(4, 4)
    expected = torch.linalg.pinv(x.reshape(-1, 4).double()) @ y.reshape(-1, 4).double()
    for block in (filters.SingleFilterBlock(4), filters.MultiLayerFilterBlock(4, 3)):
        block.init_from_pinv(x, y)
        assert torch.allclose(block(x), (x.double() @ expected).float(), atol=1e-5)
        assert (block(x) - y).square().sum() < (x - y).square().sum()


def test_filter_one_does_not_unfreeze_ten_or_eleven():
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)) for _ in range(12)])
    model.head = nn.Linear(4, 2)
    filters.freeze_non_trainable(model, [1])
    for i in (0, 10, 11):
        assert not model.blocks[i][0].weight.requires_grad
    assert model.blocks[1][0].weight.requires_grad
    assert model.blocks[10][1].weight.requires_grad
    breakdown = filters.count_parameter_breakdown(model, [1])
    assert breakdown['filter_block'] == sum(p.numel() for p in model.blocks[1].parameters())


class FakeImages(Dataset):
    classes = ['a', 'b']
    def __init__(self, transform, train):
        self.transform, self.train = transform, train
        self.image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    def __len__(self):
        return 100
    def __getitem__(self, i):
        return self.transform(self.image), i % 2


def flatten_ids(dataset):
    ids = list(range(len(dataset)))
    while isinstance(dataset, torch.utils.data.Subset):
        ids = [int(dataset.indices[i]) for i in ids]
        dataset = dataset.dataset
    return ids, dataset


@pytest.mark.parametrize('full', [False, True])
def test_validation_deterministic_split_membership_unchanged(monkeypatch, full):
    monkeypatch.setattr(data, 'get_dataset_by_name', lambda name, root, train, transform, seed: FakeImages(transform, train))
    args = SimpleNamespace(dataset='pets', batch_size=8, num_samples=40, use_full_dataset=full, seed=42)
    train, val, test, *_ = data.get_dataloaders(args)
    train_ids, train_base = flatten_ids(train.dataset)
    val_ids, val_base = flatten_ids(val.dataset)
    assert set(train_ids).isdisjoint(val_ids)
    if full:
        expected_train, expected_val = torch.utils.data.random_split(range(100), [85, 15], generator=torch.Generator().manual_seed(42))
        assert train_ids == list(expected_train.indices)
        assert val_ids == list(expected_val.indices)
    else:
        subset = torch.randperm(100, generator=torch.Generator().manual_seed(42))[:40]
        a, b = torch.utils.data.random_split(range(40), [32, 8], generator=torch.Generator().manual_seed(42))
        assert train_ids == [int(subset[i]) for i in a.indices]
        assert val_ids == [int(subset[i]) for i in b.indices]
    assert train_base is not val_base
    assert torch.equal(val.dataset[0][0], val.dataset[0][0])
    assert any(isinstance(t, data.transforms.RandomHorizontalFlip) for t in train_base.transform.transforms)
    assert not any(isinstance(t, data.transforms.RandomHorizontalFlip) for t in val_base.transform.transforms)


def test_dsprites_augmentation_preserves_pose(monkeypatch):
    monkeypatch.setattr(data, 'get_dataset_by_name', lambda name, root, train, transform, seed: FakeImages(transform, train))
    train, *_ = data.get_dataloaders(SimpleNamespace(dataset='dsprites-loc', seed=42))
    _, base = flatten_ids(train.dataset)
    assert not any(isinstance(t, data.transforms.RandomHorizontalFlip) for t in base.transform.transforms)


def load_extractor():
    tree = ast.parse((ROOT / 'train_sfp_lora.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_block_inputs_outputs')
    ns = {'torch': torch, 'nn': nn, 'DataLoader': DataLoader}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'extractor', 'exec'), ns)
    return ns[fn.name]


def test_initialization_collects_exactly_64_and_removes_hook():
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.forward = lambda x: model.blocks[0](x)
    x = torch.randn(100, 3, 4)
    loader = DataLoader(TensorDataset(x), batch_size=23)
    extract = load_extractor()
    a, b = extract(model, loader, 0, 'cpu')
    assert a.shape[0] == b.shape[0] == 64
    assert torch.equal(a, x[:64])
    assert not model.blocks[0]._forward_hooks
    with pytest.raises(ValueError, match='64 training images'):
        extract(model, DataLoader(TensorDataset(x[:20]), batch_size=7), 0, 'cpu')
    assert not model.blocks[0]._forward_hooks


def test_reporting_selects_by_validation_not_test():
    rows = [dict(dataset='pets', method='sft', block=i, seed=0, val_acc=v, test_acc=t,
                 train_compute_s=1, trainable_pct=2) for i, v, t in [(0, 90, 10), (1, 20, 99)]]
    selected = analysis.best_block_per_method(rows)['pets'][0]
    assert selected['block'] == 0
    assert selected['test_acc'] == 10


def test_preprocessing_matches_explicit_pretrained_checkpoint():
    import timm
    cfg = timm.get_pretrained_cfg('vit_base_patch16_224.augreg_in21k')
    assert cfg.num_classes == 21843
    assert tuple(data.IMAGENET_MEAN) == cfg.mean
    assert tuple(data.IMAGENET_STD) == cfg.std
