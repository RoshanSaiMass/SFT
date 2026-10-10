import random
import numpy as np
import torch
from torch.utils.data import TensorDataset

def calibration_images(loader, samples=64):
    """Cache a small training subset without advancing the main sampling RNG."""
    if samples < 1:
        raise ValueError("Calibration sample count must be positive.")
    py_state, np_state = random.getstate(), np.random.get_state()
    generator_state = loader.generator.get_state() if loader.generator is not None else None
    images, labels, seen = [], [], 0
    try:
        with torch.random.fork_rng(devices=[]):
            iterator = iter(loader)
            try:
                for batch in iterator:
                    take = min(samples - seen, len(batch[0]))
                    images.append(batch[0][:take].detach().cpu())
                    labels.append(batch[1][:take].detach().cpu())
                    seen += take
                    if seen == samples:
                        break
            finally:
                del iterator
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
        if generator_state is not None:
            loader.generator.set_state(generator_state)
    if seen == 0:
        raise ValueError("Empty training loader; cannot calibrate.")
    return TensorDataset(torch.cat(images), torch.cat(labels))
