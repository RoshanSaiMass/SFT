"""SNIP Eq. 8/9/10 candidate search; configurable protocol choices are explicit."""
import copy
from contextlib import contextmanager
import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from compressed_filters import replace_filter
from single_filter_lora import freeze_non_trainable
from train_sfp_lora import extract_block_inputs_outputs


@contextmanager
def seeded_stream(seed, loader, device):
    py_state, np_state = random.getstate(), np.random.get_state()
    state = loader.generator.get_state() if loader.generator is not None else None
    target = torch.device(device)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if loader.generator is not None:
                loader.generator.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
        if state is not None:
            loader.generator.set_state(state)


def snip_scores(model, loader, device, label_smoothing=0.0, replaced=None):
    """Sum |theta * d(mean training CE)/dtheta| across the whole network.

    Temporarily differentiate frozen weights as well, without optimizer updates.
    Microbatches accumulate gradients before the absolute product so their result
    matches one calibration loss. Existing .grad values, flags and modes survive.
    """
    named = list(model.named_parameters())
    parameters = [p for _, p in named]
    flags = [p.requires_grad for p in parameters]
    modes = {m: m.training for m in model.modules()}
    accumulated, samples, loss_sum, correct = [None] * len(parameters), 0, 0.0, 0
    try:
        for p in parameters:
            p.requires_grad_(True)
        model.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits, y, reduction="sum", label_smoothing=label_smoothing
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SNIP calibration loss")
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            with torch.no_grad():
                for i, gradient in enumerate(gradients):
                    if gradient is not None:
                        if accumulated[i] is None:
                            accumulated[i] = gradient.detach().clone()
                        else:
                            accumulated[i].add_(gradient.detach())
            samples += len(x)
            loss_sum += float(loss.detach())
            correct += int((logits.argmax(-1) == y).sum())
        if not samples:
            raise ValueError("Empty SNIP calibration loader")
        by_id = {}
        with torch.no_grad():
            for p, gradient in zip(parameters, accumulated):
                score = 0.0 if gradient is None else float((p * (gradient / samples)).abs().double().sum())
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite SNIP score")
                by_id[id(p)] = score
        blocks = {i: sum(by_id[id(p)] for p in block.parameters()) for i, block in enumerate(model.blocks)}
        ln_ids = {id(p) for module in model.modules() if isinstance(module, nn.LayerNorm)
                  for p in module.parameters()}
        components = dict(filter=0.0, layernorm=0.0, head=0.0,
                          remaining_blocks=0.0, stem_cls_position=0.0)
        for name, p in named:
            if replaced is not None and name.startswith(f"blocks.{replaced}."):
                group = "filter"
            elif id(p) in ln_ids:
                group = "layernorm"
            elif name.startswith("head.") or name.startswith("head_dist."):
                group = "head"
            elif name.startswith("blocks."):
                group = "remaining_blocks"
            else:
                group = "stem_cls_position"
            components[group] += by_id[id(p)]
        return dict(network_score=sum(by_id.values()), block_scores=blocks, samples=samples,
                    formula="sum(abs(theta * grad(mean_cross_entropy)))",
                    label_smoothing=label_smoothing, scoring_loss=loss_sum / samples,
                    scoring_accuracy=100 * correct / samples, component_scores=components,
                    trainable_snip=sum(by_id[id(p)] for p, flag in zip(parameters, flags) if flag),
                    frozen_snip=sum(by_id[id(p)] for p, flag in zip(parameters, flags) if not flag))
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
        for module, mode in modes.items():
            module.training = mode


def select_sensitive_block(scores, replaced):
    candidates = {i: score for i, score in scores.items() if i != replaced}
    if not candidates:
        raise ValueError("No surviving block available for PEFT")
    return max(candidates, key=lambda i: (candidates[i], -i))


def install_compact(model, args, index):
    return replace_filter(model, index, filter_type=args.filter_type, rank=args.filter_rank,
                          max_terms=args.symbolic_max_terms, penalty=args.symbolic_penalty,
                          search_tokens=args.symbolic_search_tokens, seed=args.seed)


def snapshot_candidate(model, index):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    prefix = f"blocks.{index}."
    # Include symbolic operator/scale buffers, while omitting unchanged dense weights.
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if name in names or name.startswith(prefix)}


def restore_candidate(base, args, index, snapshot):
    model = copy.deepcopy(base).to(args.device)
    install_compact(model, args, index)
    incompatible = model.load_state_dict(snapshot, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected candidate snapshot keys: {incompatible.unexpected_keys}")
    freeze_non_trainable(model, [index])
    return model


@torch.no_grad()
def initialize_dense_filter(block, x, y):
    """Float64 SVD with a cutoff based on the captured feature precision.

    Promoting float32 observations does not create float64-quality observations.
    This cutoff is a documented numerical choice, not a paper-specified value.
    """
    dtype = x.dtype
    a = x.reshape(-1, x.shape[-1]).double()
    b = y.reshape(-1, y.shape[-1]).double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError("Nonfinite dense-filter calibration features")
    u, singular, vh = torch.linalg.svd(a, full_matrices=False)
    rtol = max(a.shape) * torch.finfo(dtype).eps
    keep = singular > singular[0] * rtol
    rank = int(keep.sum())
    if not rank:
        raise ValueError("No feature directions survive the precision-aware pseudoinverse cutoff")
    inverse = torch.zeros_like(singular)
    inverse[keep] = singular[keep].reciprocal()
    weight = (vh.T * inverse) @ (u.T @ b)
    if not torch.isfinite(weight).all():
        raise FloatingPointError("Nonfinite dense-filter pseudoinverse weights")
    block.weight.copy_(weight.T.to(block.weight))
    block.bias.zero_()
    prediction = block(x.to(block.weight)).double()
    error = (prediction - y.to(prediction)).square()
    full_condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    block.fit_report = dict(
        type="dense", initialization="precision_aware_svd_pseudoinverse",
        source_dtype=str(dtype), solve_dtype="torch.float64", pinv_rtol=rtol,
        effective_rank=rank, feature_width=a.shape[-1], calibration_tokens=len(a),
        input_condition_number=full_condition if full_condition is None or math.isfinite(full_condition) else None,
        retained_condition_number=float(singular[0] / singular[keep][-1]),
        singular_max=float(singular[0]), singular_min=float(singular[-1]),
        weight_norm=float(block.weight.double().norm()), initialization_mse=float(error.mean()),
        cls_mse=float(error[:, 0].mean()) if x.ndim == 3 else None,
        patch_mse=float(error[:, 1:].mean()) if x.ndim == 3 and x.shape[1] > 1 else None,
        cutoff_is_paper_specified=False,
    )


def initialize_candidate(base, args, index, calibration):
    model = copy.deepcopy(base).to(args.device)
    x, y = extract_block_inputs_outputs(model, calibration, index, args.device)
    block = install_compact(model, args, index)
    if args.filter_type == "dense":
        initialize_dense_filter(block, x.to(args.device), y.to(args.device))
    else:
        block.init_from_pinv(x.to(args.device), y.to(args.device))
    freeze_non_trainable(model, [index])
    return model


def cpu_optimizer_state(optimizer):
    saved = copy.deepcopy(optimizer.state_dict())
    for state in saved["state"].values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cpu()
    return saved


def select_compact_candidate(base, train_loader, calibration_dataset, args):
    """Replace each candidate position; discard one lowest-EMA candidate per epoch.

    Every live candidate sees the same epoch's training samples and resource
    budget. SNIP fitting/scoring exclude validation and test data. Compact filters
    are this experiment's extension of the paper's affine replacement.
    """
    required_epochs = max(0, len(base.blocks) - 1)
    if args.snip_search_epochs < required_epochs:
        raise ValueError(
            f"One-candidate-per-epoch SNIP requires at least {required_epochs} "
            f"search epochs for {len(base.blocks)} positions; set --snip-search-epochs "
            f"{required_epochs} or higher."
        )
    init_data = TensorDataset(*(tensor[:64] for tensor in calibration_dataset.tensors))
    score_data = TensorDataset(*(tensor[:args.snip_samples] for tensor in calibration_dataset.tensors))
    initialization = DataLoader(init_data, batch_size=args.snip_batch_size, shuffle=False)
    scoring = DataLoader(score_data, batch_size=args.snip_batch_size, shuffle=False)
    if len(init_data) != 64:
        raise ValueError("Compact initialization requires 64 training calibration images")
    # PEFT ranking is measured on the original model, before replacement or
    # candidate fitting. It is independent of the compact filter's gradients.
    original_scores = None
    if args.adapter_type != "none":
        with seeded_stream(args.seed, train_loader, args.device):
            base.to(args.device)
            original_scores = snip_scores(base, scoring, args.device,
                                          label_smoothing=args.label_smoothing)
        original_scores["source"] = "original_vit_before_replacement_and_training"
    base = base.cpu()
    states, fit_reports, averages, history, optimizers = {}, {}, {}, [], {}
    for index in range(len(base.blocks)):
        with seeded_stream(args.seed, train_loader, args.device):
            model = initialize_candidate(base, args, index, initialization)
        states[index] = snapshot_candidate(model, index)
        fit_reports[index] = copy.deepcopy(getattr(model.blocks[index], "fit_report", {}))
        averages[index] = 0.0
        del model
    alive = list(states)
    for epoch in range(1, args.snip_search_epochs + 1):
        rows = []
        for index in alive:
            with seeded_stream(args.seed, train_loader, args.device):
                model = restore_candidate(base, args, index, states[index])
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=args.snip_search_lr, weight_decay=args.weight_decay)
            if index in optimizers:
                optimizer.load_state_dict(optimizers[index])
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            total, samples, steps = 0.0, 0, 0
            with seeded_stream(args.seed + epoch, train_loader, args.device):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(args.device), y.to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite candidate training loss")
                    loss.backward()
                    if args.grad_clip:
                        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                    optimizer.step()
                    total += float(loss.detach()) * len(x); samples += len(x); steps += 1
            if not samples:
                raise ValueError("Empty candidate training loader")
            measured = snip_scores(model, scoring, args.device,
                                   label_smoothing=args.label_smoothing, replaced=index)
            score = measured["network_score"]
            averages[index] = args.snip_momentum * averages[index] + (1 - args.snip_momentum) * score
            states[index] = snapshot_candidate(model, index)
            optimizers[index] = cpu_optimizer_state(optimizer)
            rows.append(dict(candidate_block=index, network_snip=score, ema=averages[index],
                             train_loss=total / samples, train_samples=samples, optimizer_steps=steps,
                             scoring_loss=measured.get("scoring_loss"), scoring_accuracy=measured.get("scoring_accuracy"),
                             component_scores=measured.get("component_scores"),
                             trainable_snip=measured.get("trainable_snip"), frozen_snip=measured.get("frozen_snip"),
                             filter_weight_norm=math.sqrt(sum(float(p.detach().double().square().sum())
                                                            for p in model.blocks[index].parameters()))))
            del model, optimizer
        ordered = sorted(alive, key=lambda index: (-averages[index], index))
        survivors, pruned = ordered[:-1], ordered[-1:]
        history.append(dict(epoch=epoch, candidates=rows, survived=survivors, pruned=pruned))
        print(f"[SNIP Candidates] Epoch {epoch}: survive {survivors}, prune {pruned}", flush=True)
        for index in pruned:
            del states[index]
            optimizers.pop(index, None)
        alive = survivors
        if len(alive) == 1:
            break
    if len(alive) != 1:
        raise ValueError(f"SNIP search budget exhausted with candidates {alive}; increase --snip-search-epochs.")
    index = alive[0]
    with seeded_stream(args.seed, train_loader, args.device):
        model = restore_candidate(base, args, index, states[index])
    model.blocks[index].fit_report = fit_reports[index]
    original_highest = select_sensitive_block(original_scores["block_scores"], None) if original_scores else None
    sensitive = select_sensitive_block(original_scores["block_scores"], index) if original_scores else None
    return model, dict(selection="paper-candidates", replaced_block=index, sensitive_block=sensitive,
                       target_scores=original_scores, search_history=history,
                       peft_score_source=original_scores["source"] if original_scores else None, original_highest_block=original_highest,
                       peft_target_changed_by_replacement=original_highest == index,
                       fit_report=fit_reports[index], search_epochs_applied=len(history),
                       candidate_initialization=fit_reports, implementation_version="snip_precision_aware_v4",
                       label_smoothing=args.label_smoothing,
                       scoring_gradient_scope="all_network_parameters_including_frozen_weights",
                       pruning_rule="one_lowest_network_snip_ema_candidate_per_epoch",
                       candidates_pruned_per_epoch=1, required_search_epochs=required_epochs,
                       selected_candidate_ema=averages[index], search_complete=True,
                       optimizer_continuity=True, reused_winning_candidate=True,
                       paper_equations=[8, 9, 10], init_images=64, scoring_images=len(score_data),
                       protocol_notes="Dense initialization uses a source-precision cutoff; scoring uses the training loss. "
                                      "Frozen-weight scoring scope and other search defaults are implementation choices; "
                                      "not full paper reproduction.")
