"""SNIP candidate selection for compact replacement, with PEFT only in the highest-score surviving block."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import get_dataloaders
from calibration import calibration_images
from snip_search import select_compact_candidate, install_compact
from single_filter_lora import (count_parameter_breakdown, compute_lora_orthogonality_loss,
                                inject_lora, freeze_non_trainable, resample_all_paca)
from train_sfp_lora import (set_seed, evaluate_full, build_optimizer_param_groups,
                            evaluate_and_save_misclassified)
from plotting import plot_training_curves, plot_lr_schedule, plot_param_breakdown, save_history_csv

HERE = Path(__file__).resolve().parent
BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASETS = ["pets", "svhn", "flowers102", "dtd", "caltech101", "cifar100", "fgvc_aircraft",
            "eurosat", "sun397", "pcam", "clevr", "dsprites-loc", "dsprites-ori"]
ADAPTERS = ["lora", "dora", "paca", "rpaca"]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def configuration_id(config):
    ignored = {"output_dir", "defer_test", "overwrite", "save_misclassified_images", "max_misclassified_images"}
    payload = {k: v for k, v in config.items() if k not in ignored}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, default="pets")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--use-full-dataset", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--output-dir", default=str(HERE / "outputs"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--filter-type", choices=["dense", "lowrank", "symbolic"], default="symbolic")
    p.add_argument("--filter-rank", type=int, default=16)
    p.add_argument("--symbolic-max-terms", type=int, default=2)
    p.add_argument("--symbolic-penalty", type=float, default=1e-3)
    p.add_argument("--symbolic-search-tokens", type=int, default=2048)
    p.add_argument("--snip-samples", type=int, default=64)
    p.add_argument("--snip-batch-size", type=int, default=32)
    p.add_argument("--snip-search-epochs", type=int, default=4)
    p.add_argument("--snip-search-lr", type=float, default=1e-3)
    p.add_argument("--snip-momentum", type=float, default=0.9)
    p.add_argument("--snip-prune-fraction", type=float, default=0.5)
    p.add_argument("--adapter-type", choices=ADAPTERS, default="lora")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32)
    p.add_argument("--lora-dropout", type=float, default=0)
    p.add_argument("--paca-rank", type=int, default=-1)
    p.add_argument("--paca-selection", choices=["random", "weight"], default="random")
    p.add_argument("--paca-tuner", choices=["direct", "lora", "dora"], default="direct")
    p.add_argument("--paca-adapter-rank", type=int, default=-1)
    p.add_argument("--init-method", choices=["default", "loftq"], default="default")
    p.add_argument("--loftq-bits", type=int, default=4)
    p.add_argument("--loftq-iters", type=int, default=5)
    p.add_argument("--lora-ortho-lambda1", type=float, default=0)
    p.add_argument("--lora-ortho-lambda2", type=float, default=0)
    p.add_argument("--epochs", type=int, default=-1)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0)
    p.add_argument("--grad-clip", type=float, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--defer-test", action="store_true", help="Save the checkpoint and summary without test evaluation.")
    p.add_argument("--save-misclassified-images", type=lambda x: x.lower() not in ("false", "0", "no"),
                   nargs="?", const=True, default=False)
    p.add_argument("--max-misclassified-images", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    return p


def validate(args, p=None):
    def error(message):
        if p is not None:
            p.error(message)
        raise ValueError(message)
    if args.adapter_type not in ADAPTERS:
        error("Uni-LoRA and Uni-DoRA are excluded from this experiment.")
    if not 1 <= args.filter_rank <= 768 or not 1 <= args.symbolic_max_terms <= 3:
        error("Filter rank must be 1..768 and symbolic terms 1..3.")
    if not math.isfinite(args.symbolic_penalty) or args.symbolic_penalty < 0:
        error("Symbolic penalty must be finite and nonnegative.")
    if not 0 <= args.snip_momentum < 1 or not 0 < args.snip_prune_fraction < 1:
        error("SNIP momentum must be [0,1), pruning fraction (0,1).")
    alive = 12
    for _ in range(args.snip_search_epochs):
        alive = max(1, math.ceil(alive * (1 - args.snip_prune_fraction)))
    if alive != 1:
        error("SNIP search budget must prune 12 candidates to one; increase epochs or pruning fraction.")
    if args.epochs != -1 and args.epochs <= 0:
        error("Epochs must be positive or -1 for early stopping.")
    for key in ("max_epochs", "patience", "batch_size", "num_samples", "snip_samples", "snip_batch_size", "snip_search_epochs", "symbolic_search_tokens", "lora_rank"):
        if getattr(args, key) < 1:
            error(f"{key} must be positive.")
    if args.warmup_epochs < 0 or not 0 <= args.seed < 2**32:
        error("Warmup must be nonnegative; seed must be 0..2^32-1.")
    for key in ("lr", "lora_lr", "snip_search_lr", "lora_alpha"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            error(f"{key} must be finite and positive.")
    for key in ("grad_clip", "weight_decay", "lora_ortho_lambda1", "lora_ortho_lambda2"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            error(f"{key} must be finite and nonnegative.")
    if not 0 <= args.lora_dropout < 1 or not 0 <= args.label_smoothing < 1 or not 0 <= args.min_lr_ratio <= 1:
        error("Invalid dropout, smoothing or cosine floor.")
    is_paca = args.adapter_type in ("paca", "rpaca")
    fused = is_paca and args.paca_tuner != "direct"
    cols = args.paca_rank if args.paca_rank > 0 else args.lora_rank
    if is_paca and (not 1 <= cols <= 768 or args.paca_rank not in (-1,) and args.paca_rank <= 0):
        error("PaCA column count must be 1..768; -1 means lora-rank fallback.")
    if fused and not 1 <= args.paca_adapter_rank <= cols:
        error("Fused PaCA requires 1 <= paca-adapter-rank <= columns.")
    if args.init_method == "loftq" and args.adapter_type not in ("lora", "dora"):
        error("LoftQ initialization is supported only for standalone LoRA/DoRA.")
    if not 2 <= args.loftq_bits <= 8 or args.loftq_iters < 1:
        error("LoftQ requires bits 2..8 and positive iterations.")
    if (args.lora_ortho_lambda1 or args.lora_ortho_lambda2) and (is_paca and not fused):
        error("Orthogonality is supported for standalone/fused LoRA/DoRA, not direct PaCA or Uni.")


def apply_top_peft(model, args, replaced, sensitive):
    if replaced == sensitive or not 0 <= sensitive < len(model.blocks):
        raise ValueError("PEFT target must be a surviving original block")
    excluded = set(range(len(model.blocks))) - {sensitive}
    count = inject_lora(model, excluded, args.lora_rank, args.lora_alpha, args.lora_dropout,
                        adapter_type=args.adapter_type, init_method=args.init_method,
                        loftq_bits=args.loftq_bits, loftq_iters=args.loftq_iters,
                        paca_selection=args.paca_selection, paca_tuner=args.paca_tuner,
                        paca_rank=args.paca_rank, paca_adapter_rank=args.paca_adapter_rank)
    freeze_non_trainable(model, [replaced])
    return dict(replaced_original_index=replaced, adapter_original_indices=[sensitive],
                original_block_indices=list(range(len(model.blocks))), injected_adapter_params=count,
                policy="compact + top-block PEFT + all LayerNorms + task head; dense backbone frozen")


def resample_rpaca(model, optimizer):
    for parameter in resample_all_paca(model):
        optimizer.state.pop(parameter, None)


def restore_model(summary, device=None):
    args = argparse.Namespace(**summary["config"])
    target_device = device or args.device
    model = timm.create_model(summary["backbone"], pretrained=False, num_classes=summary["num_classes"])
    model.to(target_device)
    replaced = summary["replaced_block_idx"]
    install_compact(model, args, replaced)
    apply_top_peft(model, args, replaced, summary["peft_block_idx"])
    model.load_state_dict(torch.load(summary["checkpoint_path"], map_location=target_device, weights_only=True))
    model.eval()
    return model


def evaluate_summary(path, device=None):
    """Evaluate a validation-selected candidate, without retraining."""
    path = Path(path); summary = json.loads(path.read_text())
    args = argparse.Namespace(**summary["config"])
    args.device = device or args.device
    set_seed(args.seed, deterministic=not args.fast)
    _, _, test_loader, classes, names = get_dataloaders(args)
    if classes != summary["num_classes"]:
        raise ValueError("Dataset class count changed since training.")
    model = restore_model(summary, args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.save_misclassified_images:
        loss, accuracy, saved, csv_path, image_dir = evaluate_and_save_misclassified(
            model, test_loader, args.device, criterion, str(path.parent),
            misclassified_dir_name="misclassified", class_names=names,
            max_images=args.max_misclassified_images)
        summary["misclassified_images"] = dict(saved_count=saved, csv=csv_path, dir=image_dir)
    else:
        loss, accuracy = evaluate_full(model, test_loader, args.device, criterion)
    summary.update(final_test_loss=loss, final_test_acc=accuracy, test_evaluated=True)
    write_json(path, summary)
    return summary


def train(args):
    validate(args)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback.")
    set_seed(args.seed, deterministic=not args.fast)
    config = vars(args).copy()
    ident = configuration_id(config)
    label = f"snip_compact_{args.filter_type}_{args.adapter_type}_{args.paca_tuner}_{args.dataset}_seed{args.seed}"
    output = Path(args.output_dir).resolve() / f"{label}_{ident}"
    if (output / "metrics_summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed run exists: {output}; choose another output or --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership prevents two tasks from corrupting the same run.
    import fcntl
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if (output / "metrics_summary.json").exists() and not args.overwrite:
            raise FileExistsError(f"Completed run exists: {output}; choose another output or --overwrite.")
        return _train(args, config, ident, output)
    finally:
        lock.close()


def _train(args, config, ident, output):
    started = time.perf_counter()
    train_loader, val_loader, test_loader, classes, names = get_dataloaders(args)
    base = timm.create_model(BACKBONE, pretrained=True, num_classes=classes)
    original_params = count_parameter_breakdown(base)["total_params"]
    search_start = time.perf_counter()
    cached = calibration_images(train_loader, max(64, args.snip_samples))
    model, selection = select_compact_candidate(base, train_loader, cached, args)
    search_s = time.perf_counter() - search_start
    replaced, sensitive = selection["replaced_block"], selection["sensitive_block"]
    removed_original_params = sum(p.numel() for p in base.blocks[replaced].parameters())
    structure = apply_top_peft(model, args, replaced, sensitive)
    del base
    write_json(output / "snip_search.json", selection)
    rows = [dict(block=i, score=score, compact_replaced=i == replaced, peft_target=i == sensitive)
            for i, score in selection["target_scores"]["block_scores"].items()]
    with (output / "snip_block_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"[SNIP Compact] Replaced Block(s): [{replaced}]", flush=True)
    print(f"[SNIP Compact] Highest surviving SNIP block {sensitive}: {args.adapter_type}/{args.paca_tuner} PEFT only", flush=True)
    optimizer = torch.optim.AdamW(build_optimizer_param_groups(model, args.lr, args.lora_lr),
                                 weight_decay=args.weight_decay)
    epochs = args.max_epochs if args.epochs == -1 else args.epochs
    warmup = min(args.warmup_epochs, epochs - 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    scheduler = (torch.optim.lr_scheduler.SequentialLR(optimizer,
                 [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup), cosine],
                 milestones=[warmup]) if warmup else cosine)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "lr_main", "lr_lora", "epoch_compute_s")}
    best, best_epoch, stale = -math.inf, 0, 0
    checkpoint = output / "best_model.pt"
    use_cuda = str(args.device).startswith("cuda")
    if use_cuda:
        torch.cuda.synchronize(args.device); torch.cuda.reset_peak_memory_stats(args.device)
    compute_s, training_samples = 0.0, 0
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        if args.adapter_type == "rpaca" and epoch > 1:
            resample_rpaca(model, optimizer)
        if use_cuda:
            torch.cuda.synchronize(args.device)
        tick = time.perf_counter()
        running, count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            task_loss = criterion(model(images), labels)
            loss = task_loss + compute_lora_orthogonality_loss(model, args.lora_ortho_lambda1, args.lora_ortho_lambda2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            running += float(task_loss.detach()) * len(images); count += len(images)
        if use_cuda:
            torch.cuda.synchronize(args.device)
        duration = time.perf_counter() - tick
        compute_s += duration; training_samples += count
        val_loss, val_acc = evaluate_full(model, val_loader, args.device, criterion)
        if not math.isfinite(val_loss) or not math.isfinite(val_acc):
            raise FloatingPointError("Nonfinite validation results.")
        values = dict(epoch=epoch, train_loss=running / count, val_loss=val_loss, val_acc=val_acc,
                      lr_main=optimizer.param_groups[0]["lr"], lr_lora=optimizer.param_groups[1]["lr"], epoch_compute_s=duration)
        for key, value in values.items():
            history[key].append(value)
        if val_acc > best:
            best, best_epoch, stale = val_acc, epoch, 0
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), temporary); temporary.replace(checkpoint)
        else:
            stale += 1
        scheduler.step()
        for group, peak in zip(optimizer.param_groups, (args.lr, args.lora_lr)):
            group["lr"] = max(group["lr"], peak * args.min_lr_ratio)
        print(f"Epoch {epoch}: train_loss={running/count:.4f} val_acc={val_acc:.2f}% best={best:.2f}%", flush=True)
        if args.epochs == -1 and stale >= args.patience:
            break
    train_wall_s = time.perf_counter() - training_start
    peak_allocated = torch.cuda.max_memory_allocated(args.device) / 2**20 if use_cuda else None
    peak_reserved = torch.cuda.max_memory_reserved(args.device) / 2**20 if use_cuda else None
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    breakdown = count_parameter_breakdown(model, [replaced])
    plots = plot_training_curves(history, str(output), dataset_name=args.dataset)
    plots["lr_schedule"] = plot_lr_schedule(history, str(output), dataset_name=args.dataset)
    plots["parameters"] = plot_param_breakdown(breakdown, str(output))
    history_path = save_history_csv(history, str(output))
    equation = model.blocks[replaced].equation_report() if hasattr(model.blocks[replaced], "equation_report") else {"type": "dense"}
    write_json(output / "filter_equations.json", equation)
    summary = dict(config=config, config_id=ident, backbone=BACKBONE, num_classes=classes,
                   dataset=args.dataset, seed=args.seed, method="snip-compact",
                   adapter_type=args.adapter_type, paca_tuner=args.paca_tuner,
                   mode="snip_compact_top_peft", filter_type=args.filter_type,
                   structure=structure, pruned_block_idx=[replaced], replaced_block_idx=replaced,
                   peft_block_idx=sensitive, snip_selection=selection,
                   best_val_acc=best, best_epoch=best_epoch,
                   epochs_trained=len(history["epoch"]), early_stopping_enabled=args.epochs == -1,
                   final_test_acc=None, final_test_loss=None, test_evaluated=False,
                   checkpoint_path=str(checkpoint), history_csv=history_path, plots=plots,
                   param_breakdown=breakdown, original_model_params=original_params,
                   replaced_original_params=removed_original_params,
                   total_reduction_pct=100 * (1 - breakdown["total_params"] / original_params),
                   checkpoint_bytes=checkpoint.stat().st_size,
                   efficiency=dict(snip_search_s=search_s,
                                   search_candidate_epochs=sum(len(e["candidates"]) for e in selection["search_history"]),
                                   train_compute_s=compute_s, train_wall_s=train_wall_s,
                                   throughput_samples_per_sec=training_samples / compute_s,
                                   peak_gpu_mem_allocated_mb=peak_allocated, peak_gpu_mem_reserved_mb=peak_reserved,
                                   total_elapsed_before_test_s=time.perf_counter() - started))
    if not args.defer_test:
        if args.save_misclassified_images:
            loss, accuracy, saved, csv_path, image_dir = evaluate_and_save_misclassified(
                model, test_loader, args.device, criterion, str(output), class_names=names,
                misclassified_dir_name="misclassified", max_images=args.max_misclassified_images)
            summary["misclassified_images"] = dict(saved_count=saved, csv=csv_path, dir=image_dir)
        else:
            loss, accuracy = evaluate_full(model, test_loader, args.device, criterion)
        summary.update(final_test_acc=accuracy, final_test_loss=loss, test_evaluated=True)
    path = output / "metrics_summary.json"
    write_json(path, summary)
    print(f"[SNIP Compact] Best Val Acc: {best:.2f}% (epoch {best_epoch})", flush=True)
    if summary["test_evaluated"]:
        print(f"[SNIP Compact] Final Test Acc: {summary['final_test_acc']:.2f}%", flush=True)
    print(f"Summary: {path}", flush=True)
    return path


def main():
    p = parser(); args = p.parse_args(); validate(args, p)
    train(args)


if __name__ == "__main__":
    main()
