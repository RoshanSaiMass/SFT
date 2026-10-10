"""Physical block deletion and train-only adjacent correlation/distillation."""
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from single_filter_lora import (inject_lora, freeze_non_trainable,
                                compute_lora_orthogonality_loss, resample_all_paca)


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


@torch.no_grad()
def adjacent_correlations(model, loader, device):
    """Mean per-image Pearson across flattened token/channel outputs."""
    n = len(model.blocks)
    if n < 2:
        raise ValueError("At least two blocks are required.")
    captured = {}
    def hook(index):
        def save(module, inputs, output):
            captured[index] = output.detach()
        return save
    handles = [block.register_forward_hook(hook(i)) for i, block in enumerate(model.blocks)]
    accum = [dict(pearson=0.0, valid_samples=0, cosine=0.0, rmse=0.0,
                  relative_rmse=0.0, samples=0) for _ in range(n - 1)]
    states = {module: module.training for module in model.modules()}
    model.eval()
    try:
        for images, *_ in loader:
            captured.clear(); model(images.to(device))
            for i, record in enumerate(accum):
                a, b = captured[i].flatten(1).float(), captured[i + 1].flatten(1).float()
                if a.shape != b.shape:
                    raise ValueError("Adjacent outputs must have matching token/channel dimensions.")
                ac, bc = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
                denominator = ac.norm(dim=1) * bc.norm(dim=1)
                valid = denominator > 1e-12
                corr = ((ac * bc).sum(1) / denominator.clamp_min(1e-12)).clamp(-1, 1)
                cosine = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)).clamp(-1, 1)
                rmse = (a - b).square().mean(1).sqrt()
                relative = rmse / b.square().mean(1).sqrt().clamp_min(1e-12)
                record["pearson"] += float(corr[valid].double().sum())
                record["valid_samples"] += int(valid.sum())
                for key, values in (("cosine", cosine), ("rmse", rmse), ("relative_rmse", relative)):
                    record[key] += float(values.double().sum())
                record["samples"] += len(a)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in states.items():
            module.training = training
    if not accum[0]["samples"]:
        raise ValueError("Empty calibration loader.")
    rows = []
    for i, record in enumerate(accum):
        rows.append(dict(retained_block=i, removed_block=i + 1,
                         pearson=record["pearson"] / record["valid_samples"] if record["valid_samples"] else None,
                         cosine=record["cosine"] / record["samples"],
                         rmse=record["rmse"] / record["samples"],
                         relative_rmse=record["relative_rmse"] / record["samples"],
                         samples=record["samples"], valid_samples=record["valid_samples"]))
    return rows


def select_pair(rows, threshold=0.9, pair_index=-1):
    eligible = [row for row in rows if row["pearson"] is not None and row["pearson"] >= threshold]
    if pair_index >= 0:
        eligible = [row for row in eligible if row["retained_block"] == pair_index]
    if not eligible:
        raise ValueError(f"No requested adjacent pair meets Pearson >= {threshold}. Inspect correlation CSV or lower --min-correlation.")
    return max(eligible, key=lambda row: (row["pearson"], -row["retained_block"]))


@torch.no_grad()
def capture_pair_targets(model, loader, pair_index, device):
    """Input before b1 and teacher output after b2; no teacher copy retained."""
    inputs, targets = [], []
    def before(module, args):
        inputs.append(args[0].detach().cpu())
    def after(module, args, output):
        targets.append(output.detach().cpu())
    handles = [model.blocks[pair_index].register_forward_pre_hook(before),
               model.blocks[pair_index + 1].register_forward_hook(after)]
    states = {module: module.training for module in model.modules()}
    model.eval()
    try:
        for images, *_ in loader:
            model(images.to(device))
    finally:
        for handle in handles:
            handle.remove()
        for module, training in states.items():
            module.training = training
    if not inputs:
        raise ValueError("Empty pair calibration.")
    return TensorDataset(torch.cat(inputs), torch.cat(targets))


def delete_block(model, index):
    original = list(model.blocks)
    if len(original) < 2 or not 0 <= index < len(original):
        raise ValueError("Deletion must leave at least one transformer block.")
    removed_params = sum(p.numel() for p in original[index].parameters())
    remaining = [block for i, block in enumerate(original) if i != index]
    if isinstance(model.blocks, nn.Sequential):
        model.blocks = nn.Sequential(*remaining)
    elif isinstance(model.blocks, nn.ModuleList):
        model.blocks = nn.ModuleList(remaining)
    else:
        raise TypeError("Expected Sequential or ModuleList transformer blocks.")
    return dict(removed_original_index=index, removed_original_params=removed_params,
                remaining_original_indices=[i for i in range(len(original)) if i != index])


def adapt_remaining(model, args, target_indices):
    excluded = set(range(len(model.blocks))) - set(target_indices)
    count = inject_lora(model, excluded, args.lora_rank, args.lora_alpha, args.lora_dropout,
                        adapter_type=args.adapter_type, init_method=args.init_method,
                        loftq_bits=args.loftq_bits, loftq_iters=args.loftq_iters,
                        paca_selection=args.paca_selection, paca_tuner=args.paca_tuner,
                        paca_rank=args.paca_rank, paca_adapter_rank=args.paca_adapter_rank,
                        unilora_dim=args.unilora_dim, unilora_seed=args.seed)
    # An empty filter set is crucial: surviving pretrained block weights stay frozen.
    freeze_non_trainable(model, [])
    if args.train_layernorms != "all":
        allowed = {id(module) for index in target_indices for module in model.blocks[index].modules()
                   if isinstance(module, nn.LayerNorm)}
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                keep = args.train_layernorms == "target" and (id(module) in allowed or name == "norm")
                for param in module.parameters():
                    param.requires_grad = keep
    return count


def resample_rpaca(model, optimizer):
    for param in resample_all_paca(model):
        optimizer.state.pop(param, None)


def distill_pair(block, targets, args, device):
    """Fit b1(x) toward original b2(b1(x)) with train-only cached features."""
    parameters = [p for p in block.parameters() if p.requires_grad]
    if args.distill_epochs == 0:
        return []
    if not parameters:
        raise ValueError("No trainable retained-block parameters for distillation.")
    loader = DataLoader(targets, batch_size=args.distill_batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(parameters, lr=args.distill_lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.distill_epochs):
        block.train()
        if args.adapter_type == "rpaca" and epoch:
            resample_rpaca(block, optimizer)
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (block(x) - y).square().mean() / y.square().mean().clamp_min(1e-12)
            loss = loss + compute_lora_orthogonality_loss(block, args.lora_ortho_lambda1, args.lora_ortho_lambda2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite distillation loss.")
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            total += float(loss.detach()) * len(x); seen += len(x)
        history.append(dict(epoch=epoch + 1, normalized_feature_loss=total / seen))
    return history
