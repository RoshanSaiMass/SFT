"""Paired, training-only leave-one-block-out classification-loss screening.

Deletion tolerance is a proxy for replacement tolerance, not a guarantee.
Every removal uses the same warmed-up original network and cached images.
"""
from collections import Counter, deque
import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from compressed_filters import replace_filter
from single_filter_lora import adapter_parameter_ids
from train_sfp_lora import extract_block_inputs_outputs


def install_compact(model, args, index):
    return replace_filter(model, index, filter_type=args.filter_type, rank=args.filter_rank,
                          max_terms=args.symbolic_max_terms, penalty=args.symbolic_penalty,
                          search_tokens=args.symbolic_search_tokens, seed=args.seed)


def balanced_calibration(pool, samples, seed):
    """Round-robin over observed training labels; no validation/test access.

    The cached pool is at most max(64, 4*samples) images. Coverage refers to
    this pool, not a claim that every dataset class has been observed.
    """
    if samples < 1 or not len(pool):
        raise ValueError("Positive sample count and nonempty training pool required")
    images, labels = pool.tensors
    generator = torch.Generator().manual_seed(seed)
    buckets = {}
    for index in torch.randperm(len(pool), generator=generator).tolist():
        buckets.setdefault(int(labels[index]), deque()).append(index)
    classes = sorted(buckets)
    order = [classes[i] for i in torch.randperm(len(classes), generator=generator).tolist()]
    selected = []
    while len(selected) < min(samples, len(pool)):
        for label in order:
            if buckets[label]:
                selected.append(buckets[label].popleft())
            if len(selected) == min(samples, len(pool)):
                break
    indices = torch.tensor(selected)
    subset = TensorDataset(images[indices], labels[indices])
    counts = dict(sorted(Counter(int(y) for y in subset.tensors[1]).items()))
    report = dict(source="training_split_only", pool_samples=len(pool), scoring_samples=len(subset),
                  sampling="seeded_round_robin_over_observed_training_labels",
                  pool_class_counts=dict(sorted(Counter(int(y) for y in labels).items())),
                  scoring_class_counts=counts, selected_pool_indices=selected,
                  seed=seed, cached_images_reused_for_every_candidate=True)
    return subset, report


def warmup_head(model, loader, args):
    """Train only the target head, once; retain that head for all removals/final training."""
    started = time.perf_counter()
    head = getattr(model, "head", None)
    if head is None or not list(head.parameters()):
        raise ValueError("LOO requires a trainable target-task classifier head")
    flags = {p: p.requires_grad for p in model.parameters()}
    modes = {module: module.training for module in model.modules()}
    history = []
    try:
        model.requires_grad_(False)
        head.requires_grad_(True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_warmup_lr,
                                     weight_decay=args.weight_decay)
        for epoch in range(1, args.head_warmup_epochs + 1):
            # Frozen feature extractor, including LayerNorms, stays in evaluation mode.
            model.eval(); head.train()
            loss_sum, correct, count = 0.0, 0, 0
            for x, y in loader:
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite classifier warm-up loss")
                loss.backward(); optimizer.step()
                count += len(y); loss_sum += float(loss.detach()) * len(y)
                correct += int((logits.argmax(-1) == y).sum())
            if not count:
                raise ValueError("Empty classifier warm-up training loader")
            row = dict(epoch=epoch, loss=loss_sum/count, accuracy=100*correct/count, samples=count)
            history.append(row)
            print(f"[LOO Head Warmup] Epoch {epoch}: loss={row['loss']:.4f} accuracy={row['accuracy']:.2f}%", flush=True)
        optimizer.zero_grad(set_to_none=True)
    finally:
        for parameter, flag in flags.items():
            parameter.requires_grad_(flag)
        for module, mode in modes.items():
            module.training = mode
    return dict(epochs=args.head_warmup_epochs, lr=args.head_warmup_lr,
                trainable_head_params=sum(p.numel() for p in head.parameters()),
                backbone_updated=False, classifier_retained=True, history=history,
                wall_s=time.perf_counter()-started)


@torch.no_grad()
def screen_removals(model, data, batch_size, device, label_smoothing=0.0):
    """Measure signed delta CE on identical cached images, restoring the exact model.

    Original indices are preserved in the result. No cumulative deletion and
    no optimizer, recovery training, or gradient computation occurs here.
    """
    original = model.blocks
    blocks = list(original)
    if len(blocks) < 2 or not isinstance(original, (nn.Sequential, nn.ModuleList)):
        raise ValueError("LOO requires at least two blocks in Sequential or ModuleList")
    if not len(data) or batch_size < 1:
        raise ValueError("Empty LOO calibration data or invalid batch size")
    modes = {module: module.training for module in model.modules()}
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))

    def measure():
        loss_sum, correct, count = 0.0, 0, 0
        for x, y in loader:
            y = y.to(device); logits = model(x.to(device))
            losses = F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite LOO calibration loss")
            loss_sum += float(losses.double().sum())
            count += len(y); correct += int((logits.argmax(-1) == y).sum())
        return loss_sum/count, 100*correct/count

    try:
        model.eval()
        baseline_loss, baseline_acc = measure()
        rows = []
        for index in range(len(blocks)):
            # Rebuild from the intact original list on EVERY iteration.
            remaining = [b for i, b in enumerate(blocks) if i != index]
            model.blocks = (nn.Sequential(*remaining) if isinstance(original, nn.Sequential)
                            else nn.ModuleList(remaining))
            loss, accuracy = measure()
            rows.append(dict(block=index, loss=loss, loss_increase=loss-baseline_loss,
                             accuracy=accuracy, accuracy_drop=baseline_acc-accuracy,
                             baseline_loss=baseline_loss, baseline_accuracy=baseline_acc,
                             samples=len(data)))
    finally:
        model.blocks = original
        for module, mode in modes.items():
            module.training = mode
    return dict(baseline_loss=baseline_loss, baseline_accuracy=baseline_acc, rows=rows,
                screening_forward_images=(len(blocks)+1)*len(data))


def choose_blocks(rows):
    """Minimum signed delta for replacement; maximum surviving delta for PEFT."""
    if len(rows) < 2 or sorted(r["block"] for r in rows) != list(range(len(rows))):
        raise ValueError("Complete independent LOO rows with unique original indices required")
    if any(not math.isfinite(r["loss_increase"]) for r in rows):
        raise ValueError("LOO losses must be finite")
    replaced = min(rows, key=lambda r: (r["loss_increase"], r["block"]))["block"]
    highest = max(rows, key=lambda r: (r["loss_increase"], -r["block"]))["block"]
    sensitive = max((r for r in rows if r["block"] != replaced),
                    key=lambda r: (r["loss_increase"], -r["block"]))["block"]
    return replaced, sensitive, highest


def assert_trainable_scope(model, replaced, sensitive):
    """Fail before training if anything outside filter/PEFT/LN/head is trainable."""
    adapters = adapter_parameter_ids(model)
    allowed = {id(p) for p in model.blocks[replaced].parameters()}
    allowed.update(id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters())
    allowed.update(id(p) for p in model.head.parameters())
    allowed.update(adapters)
    if {id(p) for p in model.parameters() if p.requires_grad} != allowed:
        raise RuntimeError("Unexpected trainable parameter scope")
    if sensitive is None:
        if adapters:
            raise RuntimeError("No-PEFT run unexpectedly has adapters")
    elif replaced == sensitive or adapters != adapter_parameter_ids(model.blocks[sensitive]):
        raise RuntimeError("PEFT must occur only in the selected surviving original block")
    return [name for name, p in model.named_parameters() if p.requires_grad]


def select_compact_candidate(base, train_loader, pool, args):
    if len(pool) < 64:
        raise ValueError("Compact filter initialization requires 64 training images")
    base.to(args.device)
    score_data, calibration = balanced_calibration(pool, args.loo_samples, args.seed)
    warmup = warmup_head(base, train_loader, args)
    if args.head_warmup_epochs == 0:
        print("[LOO Warning] Head warm-up disabled: ranking uses an untrained target head.", flush=True)
    observed = set(int(y) for y in score_data.tensors[1])
    classes = getattr(base, "num_classes", None)
    missing = sorted(set(range(classes))-observed) if classes is not None else []
    calibration.update(task_class_count=classes, missing_task_classes=missing)
    if missing:
        print(f"[LOO Warning] Scoring subset misses {len(missing)} task classes; consider increasing --loo-samples.", flush=True)
    started = time.perf_counter()
    result = screen_removals(base, score_data, args.loo_batch_size, args.device, args.label_smoothing)
    screening_s = time.perf_counter()-started
    replaced, sensitive, highest = choose_blocks(result["rows"])
    removed_params = sum(p.numel() for p in base.blocks[replaced].parameters())
    for row in result["rows"]:
        row.update(compact_replaced=row["block"] == replaced,
                   peft_target=args.adapter_type != "none" and row["block"] == sensitive)
        print(f"[LOO] Block {row['block']}: loss={row['loss']:.6f} delta={row['loss_increase']:+.6f}", flush=True)
    started = time.perf_counter()
    init_data = TensorDataset(*(tensor[:64] for tensor in pool.tensors))
    init_loader = DataLoader(init_data, batch_size=args.loo_batch_size,
                             generator=torch.Generator().manual_seed(args.seed))
    x, y = extract_block_inputs_outputs(base, init_loader, replaced, args.device)
    block = install_compact(base, args, replaced)
    block.init_from_pinv(x.to(args.device), y.to(args.device))
    result.update(replaced_block=replaced, sensitive_block=sensitive if args.adapter_type != "none" else None,
                  original_highest_block=highest, peft_target_changed_by_replacement=highest == replaced and args.adapter_type != "none",
                  replaced_original_params=removed_params, head_warmup=warmup,
                  calibration=calibration, init_images=64, init_source="training_pool_first_64",
                  screening_s=screening_s, filter_initialization_s=time.perf_counter()-started,
                  ranking="signed mean CE loss increase versus intact warmed-up original ViT",
                  label_smoothing=args.label_smoothing,
                  replacement_rule="minimum_delta_loss_lowest_original_index_on_ties",
                  peft_rule="maximum_surviving_delta_loss_lowest_original_index_on_ties",
                  score_source="original_vit_before_replacement_and_recovery_training",
                  independent_removals=True, deletion_is_screening_only=True,
                  used_validation=False, used_test=False)
    return base, result
