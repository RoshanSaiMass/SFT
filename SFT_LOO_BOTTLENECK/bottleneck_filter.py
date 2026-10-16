"""Tokenwise residual GELU bottleneck, fitted only to training activations."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class BottleneckFilterBlock(nn.Module):
    def __init__(self, dim, rank, search_tokens=2048, seed=18, init_steps=200, init_lr=1e-3):
        super().__init__()
        if not 1 <= rank <= dim or search_tokens < 2 or init_steps < 0:
            raise ValueError('Invalid bottleneck width, token budget or initialization steps')
        self.rank, self.search_tokens, self.seed = rank, search_tokens, seed
        self.init_steps, self.init_lr = init_steps, init_lr
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.fit_report = {}

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.norm(x))))

    def init_from_pinv(self, inputs, outputs):
        """Image-disjoint fit/holdout; PCA start, linear solve, then local reconstruction.

        No validation/test images, classifier loss or original block parameters are
        used in this initializer. Best holdout state (including step zero) wins.
        """
        if inputs.shape != outputs.shape or inputs.ndim != 3 or len(inputs) < 2:
            raise ValueError('Need paired [images,tokens,channels] with at least two images')
        device, dtype = self.up.weight.device, self.up.weight.dtype
        split = max(1, min(len(inputs) - 1, int(0.75 * len(inputs))))
        generator = torch.Generator().manual_seed(self.seed)
        def sample(x, y):
            x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
            ix = torch.randperm(len(x), generator=generator)[:self.search_tokens].to(x.device)
            return x[ix].to(device=device, dtype=dtype), y[ix].to(device=device, dtype=dtype)
        x, y = sample(inputs[:split], outputs[:split])
        vx, vy = sample(inputs[split:], outputs[split:])
        with torch.no_grad():
            z = self.norm(x).detach().double().cpu()
            # Deterministic right singular vectors of the normalized token matrix.
            _, _, vh = torch.linalg.svd(z, full_matrices=False)
            n = min(self.rank, len(vh))
            self.down.weight[:n].copy_(vh[:n].to(device=device, dtype=dtype))
            self.down.bias.zero_()
            hidden = F.gelu(self.down(self.norm(x))).double().cpu()
            design = torch.cat([hidden, torch.ones(len(hidden), 1, dtype=torch.float64)], dim=1)
            target = (y - x).double().cpu()
            solution = torch.linalg.lstsq(design, target, driver='gelsd').solution
            self.up.weight.copy_(solution[:-1].T.to(device=device, dtype=dtype))
            self.up.bias.copy_(solution[-1].to(device=device, dtype=dtype))
        scale = (y - x).square().mean().detach().clamp_min(1e-8)
        def heldout():
            with torch.no_grad():
                return float(F.mse_loss(self(vx), vy))
        initial = best = heldout()
        best_state = copy.deepcopy(self.state_dict())
        best_step = 0
        # Activations are detached: gradients never reach the teacher backbone.
        with torch.enable_grad():
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.init_lr, weight_decay=0.0)
            gen = torch.Generator().manual_seed(self.seed + 1)
            for step in range(1, self.init_steps + 1):
                ix = torch.randperm(len(x), generator=gen)[:256].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(self(x[ix].detach()), y[ix].detach()) / scale
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite bottleneck reconstruction loss')
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                value = heldout()
                if not torch.isfinite(torch.tensor(value)):
                    raise FloatingPointError('Nonfinite bottleneck holdout reconstruction loss')
                if value < best:
                    best, best_step = value, step
                    best_state = copy.deepcopy(self.state_dict())
            optimizer.zero_grad(set_to_none=True)
        self.load_state_dict(best_state)
        self.fit_report = dict(initial_holdout_mse=initial, best_holdout_mse=best,
                               best_step=best_step, steps=self.init_steps,
                               fit_images=split, holdout_images=len(inputs)-split,
                               fit_tokens=len(x), holdout_tokens=len(vx),
                               source='training_images_only', image_disjoint=True,
                               initialization='PCA_down_GELU_lstsq_up_then_reconstruction')

    def equation_report(self):
        return dict(type='residual_gelu_bottleneck', rank=self.rank,
                    equation='y = x + up(GELU(down(LayerNorm(x))))',
                    parameter_count=sum(p.numel() for p in self.parameters()),
                    fit_report=self.fit_report)


def replace_filter(model, index, rank, **kwargs):
    original = next(model.blocks[index].parameters())
    dim = getattr(model, 'embed_dim', model.num_features)
    block = BottleneckFilterBlock(dim, rank, **kwargs).to(device=original.device, dtype=original.dtype)
    model.blocks[index] = block
    return block
