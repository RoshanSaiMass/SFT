"""Dataset startup regressions without downloading benchmark data."""

from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch.utils.data import Dataset, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data


class FakeCaltech101(Dataset):
    # Torchvision's Caltech101 uses categories and can return grayscale images.
    categories = [str(i) for i in range(101)]

    def __init__(self, root, download, transform):
        self.transform = transform

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        return self.transform(Image.new("L", (32, 32), 128)), 100


def source_indices(dataset):
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [int(dataset.indices[i]) for i in indices]
        dataset = dataset.dataset
    return indices


@pytest.mark.parametrize("full", [False, True])
def test_caltech_class_count_grayscale_and_original_split_indices(monkeypatch, tmp_path, full):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data.datasets, "Caltech101", FakeCaltech101)
    args = SimpleNamespace(dataset="caltech101", batch_size=2, num_samples=100,
                           use_full_dataset=full, seed=42)
    train, val, test, num_classes, names = data.get_dataloaders(args)
    assert num_classes == 101
    assert names == FakeCaltech101.categories
    for loader in (train, val, test):
        image, label = loader.dataset[0]
        assert image.shape == (3, 224, 224)
        loss = torch.nn.functional.cross_entropy(torch.zeros(1, num_classes), torch.tensor([label]))
        assert torch.isfinite(loss)

    # Recreate the original two-stage Caltech split with independent generators.
    expected_pool, expected_test = random_split(
        range(1000), [800, 200], generator=torch.Generator().manual_seed(42))
    if full:
        expected_train, expected_val = random_split(
            expected_pool, [680, 120], generator=torch.Generator().manual_seed(42))
    else:
        chosen = torch.randperm(800, generator=torch.Generator().manual_seed(42))[:100]
        expected_train, expected_val = random_split(
            Subset(expected_pool, chosen), [80, 20], generator=torch.Generator().manual_seed(42))
    assert source_indices(train.dataset) == source_indices(expected_train)
    assert source_indices(val.dataset) == source_indices(expected_val)
    assert source_indices(test.dataset) == source_indices(expected_test)


def test_rgb_transform_can_be_pickled_for_spawn_workers():
    transform = pickle.loads(pickle.dumps(data.ensure_rgb))
    assert transform(Image.new("L", (4, 4))).mode == "RGB"
