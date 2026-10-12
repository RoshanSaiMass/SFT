"""Pseudoinverse compression and bounded symbolic regression for token features.

Equation search uses a held-out set of calibration *images*, all from training.
This is a small, explicit expression library, not unrestricted genetic search.
"""
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

OPERATORS = ("linear", "square", "cube", "tanh", "sin", "constant")


def basis(z, operator):
    if operator == "linear":
        return z
    if operator == "square":
        return z.clamp(-3, 3).square()
    if operator == "cube":
        return z.clamp(-3, 3).pow(3)
    if operator == "tanh":
        return z.tanh()
    if operator == "sin":
        return z.sin()
    if operator == "constant":
        return torch.ones_like(z)
    raise ValueError(operator)


def flattened(x, y):
    if x.shape != y.shape or x.ndim < 2:
        raise ValueError("Calibration input/output shapes must match and be at least 2-D.")
    return x.reshape(-1, x.shape[-1]).double(), y.reshape(-1, y.shape[-1]).double()


def coefficients(z, target, terms):
    # Independent coefficients per latent coordinate; a shared expression structure.
    features = torch.stack([basis(z, op) for op in terms], dim=-1)
    return (torch.linalg.pinv(features.transpose(0, 1)) @
            target.T.unsqueeze(-1)).squeeze(-1).T


def predict_latent(z, coeff, terms):
    return sum(basis(z, term) * coeff[i] for i, term in enumerate(terms))


class LowRankFilterBlock(nn.Module):
    """y = U(Vx) + b, initialized by truncated SVD of X^+Y."""
    def __init__(self, embed_dim, rank=16, dropout=0.0):
        super().__init__()
        if not 1 <= rank <= embed_dim:
            raise ValueError("Filter rank must be between 1 and embedding dimension.")
        self.embed_dim, self.rank = embed_dim, rank
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.fit_report = {}

    @torch.no_grad()
    def init_from_pinv(self, x, y):
        a, b = flattened(x, y)
        weight = (torch.linalg.pinv(a) @ b).T
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        root = s[:self.rank].sqrt()
        self.down.weight.copy_((root[:, None] * vh[:self.rank]).to(self.down.weight))
        self.up.weight.copy_((u[:, :self.rank] * root).to(self.up.weight))
        self.up.bias.zero_()
        approximation = (u[:, :self.rank] * s[:self.rank]) @ vh[:self.rank]
        energy = s.square().sum().clamp_min(torch.finfo(s.dtype).eps)
        self.fit_report = {
            "calibration_tokens": len(a),
            "retained_matrix_energy": float(s[:self.rank].square().sum() / energy),
            "initialization_mse": float((a @ approximation.T - b).square().mean()),
            "dense_initialization_mse": float((a @ weight.T - b).square().mean()),
        }

    def forward(self, x):
        return self.dropout(self.up(self.down(x)))

    def equation_report(self):
        return {"type": "lowrank", "rank": self.rank,
                "equation": "y = linear(linear(x, V, no_bias), U, bias)", "initialization": self.fit_report}


class SymbolicFilterBlock(nn.Module):
    """y = x + up(scale * sum_t c[t] * f_t(down(x)/scale)) + bias.

    Pseudoinverse/SVD fits a low-rank residual basis. Held-out training images
    select a sparse equation structure; its coefficients are then refitted on
    all calibration images. Projections and coefficients remain trainable.
    """
    def __init__(self, embed_dim, rank=16, max_terms=2, penalty=1e-3,
                 search_tokens=2048, seed=18, dropout=0.0):
        super().__init__()
        if not 1 <= rank <= embed_dim:
            raise ValueError("Filter rank must be between 1 and embedding dimension.")
        if not 1 <= max_terms <= 3 or penalty < 0 or search_tokens < 1:
            raise ValueError("Require 1..3 terms, nonnegative penalty, positive search token cap.")
        self.embed_dim, self.rank = embed_dim, rank
        self.max_terms, self.penalty = max_terms, penalty
        self.search_tokens, self.seed = search_tokens, seed
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=True)
        self.coeff = nn.Parameter(torch.zeros(max_terms, rank))
        # Fixed shape makes checkpoint loading independent of selected term count.
        self.register_buffer("operator_ids", torch.full((max_terms,), -1, dtype=torch.long))
        self.register_buffer("scale", torch.ones(rank))
        self._active_terms = ()
        self.register_load_state_dict_post_hook(self._refresh_operators)
        self.dropout = nn.Dropout(dropout)
        self.fit_report = {}
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _refresh_operators(self, *unused):
        # Refresh once after fitting/loading, avoiding a CUDA-to-host sync in
        # every forward pass. Operator IDs remain checkpointed as buffers.
        self._active_terms = tuple((row, OPERATORS[code]) for row, code in
                                   enumerate(self.operator_ids.tolist()) if code >= 0)

    def _subsample(self, x, y, generator):
        if len(x) > self.search_tokens:
            idx = torch.randperm(len(x), generator=generator)[:self.search_tokens].to(x.device)
            x, y = x[idx], y[idx]
        return x, y

    @torch.no_grad()
    def init_from_pinv(self, x, y):
        if x.shape != y.shape or x.shape[0] < 4:
            raise ValueError("Symbolic fitting requires matching features from at least four images.")
        g = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(x.shape[0], generator=g).to(x.device)
        split = max(1, min(x.shape[0] - 1, int(0.75 * x.shape[0])))
        fit_images, check_images = order[:split], order[split:]
        a, b = flattened(x[fit_images], y[fit_images])
        c, d = flattened(x[check_images], y[check_images])
        a, b = self._subsample(a, b, g)
        c, d = self._subsample(c, d, g)
        # Row-vector convention: residual ~= x @ projection @ reconstruction.
        residual_map = torch.linalg.pinv(a) @ (b - a)
        u, singular, vh = torch.linalg.svd(residual_map, full_matrices=False)
        projection = u[:, :self.rank] * singular[:self.rank]
        reconstruction = vh[:self.rank]
        scale = (a @ projection).square().mean(0).sqrt().clamp_min(1e-6)
        z, zv = (a @ projection) / scale, (c @ projection) / scale
        target = ((b - a) @ reconstruction.T) / scale
        validation_energy = (d - c).square().mean().clamp_min(1e-12)
        best, evaluations = None, []
        for count in range(1, self.max_terms + 1):
            for terms in combinations(OPERATORS, count):
                fitted = coefficients(z, target, terms)
                prediction = c + (predict_latent(zv, fitted, terms) * scale) @ reconstruction
                mse = float((prediction - d).square().mean())
                objective = mse / float(validation_energy) + self.penalty * count
                evaluations.append({"terms": list(terms), "heldout_mse": mse,
                                    "selection_objective": objective})
                if best is None or objective < best[0]:
                    best = (objective, terms, fitted, mse)
        _, terms, _, heldout_mse = best
        # Refit only coefficients with the selected structure/basis fixed.
        all_x, all_y = flattened(x, y)
        all_x, all_y = self._subsample(all_x, all_y, g)
        fitted = coefficients((all_x @ projection) / scale,
                              ((all_y - all_x) @ reconstruction.T) / scale, terms)
        self.down.weight.copy_(projection.T.to(self.down.weight))
        self.up.weight.copy_(reconstruction.T.to(self.up.weight))
        self.up.bias.zero_()
        self.scale.copy_(scale.to(self.scale))
        self.coeff.zero_()
        self.coeff[:len(terms)].copy_(fitted.to(self.coeff))
        self.operator_ids.fill_(-1)
        self.operator_ids[:len(terms)] = torch.tensor(
            [OPERATORS.index(t) for t in terms], device=self.operator_ids.device)
        self._refresh_operators()
        was_training = self.training
        self.eval()
        initialization_mse = float((self(all_x.to(self.down.weight)) -
                                    all_y.to(self.down.weight)).square().mean())
        self.train(was_training)
        self.fit_report = {
            "search_split": "75/25 by training calibration image; no validation/test images",
            "fit_images": len(fit_images), "heldout_images": len(check_images),
            "fit_tokens": len(a), "heldout_tokens": len(c), "refit_tokens": len(all_x),
            "selected_terms": list(terms), "heldout_mse_before_refit": heldout_mse,
            "identity_heldout_mse": float(validation_energy), "candidates": evaluations,
            "initialization_mse_after_refit": initialization_mse,
        }

    def forward(self, x):
        z = self.down(x) / self.scale
        latent = torch.zeros_like(z)
        for row, operator in self._active_terms:
            latent = latent + self.coeff[row] * basis(z, operator)
        return self.dropout(x + self.up(latent * self.scale))

    def equation_report(self):
        terms = [OPERATORS[code] for code in self.operator_ids.tolist() if code >= 0]
        return {"type": "symbolic", "rank": self.rank, "terms": terms,
                "equation": "z=linear(x,V,no_bias)/scale; y=x+linear(scale*sum_t coeff[t]*f_t(z),U,bias)",
                "square_cube_domain": "clamp(z, -3, 3) before power",
                "coefficients": self.coeff.detach().cpu().tolist(),
                "scale": self.scale.detach().cpu().tolist(),
                "projection_storage": "down/up matrices are in best checkpoint and counted",
                "initialization": self.fit_report}


def replace_filter(model, index, *, filter_type="dense", rank=16, max_terms=2,
                   penalty=1e-3, search_tokens=2048, seed=18, **baseline_options):
    from single_filter_lora import make_filter_block
    reference = next(model.blocks[index].parameters())
    width = getattr(model, "embed_dim", reference.shape[-1])
    if filter_type not in ("dense", "lowrank", "symbolic"):
        raise ValueError(f"Unknown filter type: {filter_type}")
    # Constructors use CPU random initialization, then inheritance overwrites it.
    # Preserve the loader/augmentation RNG stream across filter architectures.
    with torch.random.fork_rng(devices=[]):
        if filter_type == "dense":
            block = make_filter_block(embed_dim=width, **baseline_options)
        else:
            if baseline_options.get("num_layers", 1) != 1 or baseline_options.get("residual_hidden_dim", 0):
                raise ValueError("Compressed filters require one layer and no extra dense residual.")
            options = dict(embed_dim=width, rank=rank, dropout=baseline_options.get("dropout", 0.0))
            block = (LowRankFilterBlock(**options) if filter_type == "lowrank" else
                     SymbolicFilterBlock(**options, max_terms=max_terms, penalty=penalty,
                                         search_tokens=search_tokens, seed=seed))
    # Sequential multiple replacement must keep new filters on the model's device.
    model.blocks[index] = block.to(device=reference.device, dtype=reference.dtype)
    return model.blocks[index]
