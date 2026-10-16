import errno
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_bottleneck as train


@pytest.mark.parametrize('init_method', ['default', 'loftq'])
def test_compact_checkpoint_preserves_trainable_buffers_and_loftq_base(tmp_path, init_method):
    model = torch.nn.Module()
    model.num_classes = 2
    model.frozen = torch.nn.Linear(100, 100)
    model.frozen.requires_grad_(False)
    model.adapter = torch.nn.Module()
    model.adapter.base_layer = torch.nn.Linear(2, 2)
    model.adapter.base_layer.requires_grad_(False)
    model.head = torch.nn.Linear(2, 2)
    model.register_buffer('operator_ids', torch.tensor([1, 6, 8]))
    original = {k: v.clone() for k, v in model.state_dict().items()}
    payload = train.checkpoint_payload(model, SimpleNamespace(checkpoint_format='trainable', init_method=init_method))
    assert 'frozen.weight' not in payload['state_dict']
    assert ('adapter.base_layer.weight' in payload['state_dict']) == (init_method == 'loftq')
    path = tmp_path / 'compact.pt'
    train.save_checkpoint_verified(payload, path)
    with torch.no_grad():
        model.head.weight.zero_()
        model.operator_ids.zero_()
        if init_method == 'loftq':
            model.adapter.base_layer.weight.zero_()
    train.load_checkpoint_state(model, torch.load(path, weights_only=True))
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    del payload['state_dict']['head.weight']
    with pytest.raises(RuntimeError, match='missing required state'):
        train.load_checkpoint_state(model, payload)


def test_verified_checkpoint_can_be_read_after_publication(tmp_path):
    path = tmp_path / 'best_model.pt'
    state = {'weight': torch.randn(3, 2), 'operator_ids': torch.tensor([0, 6, 8])}
    train.save_checkpoint_verified(state, path)
    actual = torch.load(path, map_location='cpu', weights_only=True)
    assert all(torch.equal(actual[name], value) for name, value in state.items())
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', ['empty', 'truncated', 'no_space', 'sync_error'])
def test_bad_write_is_detected_before_replacing_previous_best(tmp_path, monkeypatch, failure):
    path = tmp_path / 'best_model.pt'
    state = {'weight': torch.ones(3, 2)}
    train.save_checkpoint_verified(state, path)
    previous = path.read_bytes()
    def bad_save(*args, **kwargs):
        if failure == 'truncated':
            args[1].write(b'PK\x03\x04broken')
        elif failure == 'no_space':
            raise OSError(errno.ENOSPC, 'No space left on device')
    if failure == 'sync_error':
        def bad_sync(*args):
            raise OSError(errno.EIO, 'Input/output error')
        monkeypatch.setattr(train.os, 'fsync', bad_sync)
    else:
        monkeypatch.setattr(train.torch, 'save', bad_save)
    with pytest.raises(RuntimeError, match='Checkpoint save/verification failed'):
        train.save_checkpoint_verified({'weight': torch.zeros(3, 2)}, path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp'))
