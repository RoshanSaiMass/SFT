"""
Diagnostic: measures how much each LoRA adapter actually learned, by comparing the
magnitude of its effective weight delta (scaling * B @ A) against the magnitude of
the frozen base layer weight it's attached to.

Answers: "did the LoRA adapters wake up and learn something substantial during
training, or did they stay close to their zero-initialized starting point?"
This disambiguates two very different explanations for LoRA not improving accuracy:
  - tiny deltas  -> adapters never really trained (short schedule / low effective LR
    for the LoRA path / undertrained), not a lack-of-capacity-being-useful problem
  - large deltas -> adapters ARE learning substantial updates, so if accuracy still
    isn't improving, the issue is more likely overfitting / wrong placement / the
    filter block already saturating what's learnable from this data, not "LoRA
    failed to train at all"

Usage:
    python3 lora_delta_diagnostic.py <run_output_dir>

<run_output_dir> must be the output directory of a --lora-rank > 0 training run,
containing metrics_summary.json and the saved best-checkpoint .pt file.
Also supports DoRA, direct/fused PaCA/RPaCA and Uni-LoRA/Uni-DoRA using saved metadata. Reuses
substitute_filter_block() / inject_lora() / freeze_non_trainable() from
single_filter_lora.py to reconstruct the exact same wrapped architecture that was
trained (supports single-block and multi-block --num-filter-blocks runs,
--filter-block-layers 1 and >1, and --filter-residual-hidden-dim 0 and >0), so the
checkpoint loads correctly.

Handles both current metrics_summary.json format (pruned_block_idx as a list) and
older formats (pruned_block_idx as a single int, filter_block_layers /
filter_residual_hidden_dim absent -> default to 1 / 0 respectively, matching what
existed before those fields were added).

ASSUMPTION: reconstructs using the default target_keywords ["qkv", "proj", "fc1",
"fc2"] -- this matches every run produced by the current train_sfp_lora.py (that
parameter isn't exposed as a CLI flag / recorded in metrics_summary.json, so this
script assumes it wasn't changed between training and now).
"""

import argparse
import json
import os

import timm
import torch

from single_filter_lora import (LoRALinear, PaCALinear, PaCAAdapterLinear, UniLoRAAdapterLinear,
                                substitute_filter_block, inject_lora, freeze_non_trainable)


def load_run_model(output_dir: str, device: str = "cpu"):
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"metrics_summary.json not found in {output_dir}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    if summary.get("full_finetune"):
        raise ValueError("This run used --full-finetune -- there are no adaptation modules to inspect.")

    lora_rank = summary.get("lora_rank")
    adapter_type = summary.get("adapter_type") or "lora"
    paca_columns = summary.get("paca_columns") or lora_rank or 0
    if (adapter_type not in ("paca", "rpaca") and (not lora_rank or lora_rank <= 0)) or (
        adapter_type in ("paca", "rpaca") and paca_columns <= 0
    ):
        raise ValueError("This run contains no adaptation modules to inspect.")

    # pruned_block_idx is a list in current metrics_summary.json (multi-block support),
    # but was a single int in older summaries -- normalize to a list either way.
    raw_pruned = summary["pruned_block_idx"]
    pruned_block_indices = [raw_pruned] if isinstance(raw_pruned, int) else list(raw_pruned)

    # filter_block_layers wasn't recorded in older summaries -- default to 1
    # (the only option that existed before this field was added).
    filter_block_layers = summary.get("filter_block_layers", 1)

    # filter_residual_hidden_dim wasn't recorded in older summaries either --
    # default to 0 (no residual branch, the only option that existed before this
    # field was added). Must match what was actually trained with, or
    # load_state_dict below will fail (missing/unexpected keys for the residual
    # branch's fc1/fc2 params).
    filter_residual_hidden_dim = summary.get("filter_residual_hidden_dim", 0) or 0
    filter_residual_alpha = summary.get("filter_residual_alpha", 1.0)
    if filter_residual_alpha is None:
        filter_residual_alpha = 1.0
    filter_residual_dropout = summary.get("filter_residual_dropout") or 0.0

    lora_alpha = summary.get("lora_alpha", 32.0)
    lora_dropout = summary.get("lora_dropout", 0.0) or 0.0
    num_classes = summary["num_classes"]
    checkpoint_path = summary.get("checkpoint_path") or os.path.join(
        output_dir, f"best_sfp_lora_{summary['dataset']}.pt"
    )
    if not os.path.isfile(checkpoint_path):
        local_checkpoint = os.path.join(output_dir, os.path.basename(checkpoint_path))
        if os.path.isfile(local_checkpoint):
            checkpoint_path = local_checkpoint
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Diagnostic] Rebuilding architecture: pruned_block_indices={pruned_block_indices}, "
          f"filter_block_layers={filter_block_layers}, "
          f"filter_residual_hidden_dim={filter_residual_hidden_dim}, "
          f"lora_rank={lora_rank}, lora_alpha={lora_alpha}, "
          f"lora_dropout={lora_dropout}, num_classes={num_classes}")

    # pretrained=False is safe AND faster here: every weight gets immediately
    # overwritten by the checkpoint's state_dict below, so the initial pretrained
    # weights are never actually used -- no need to download them.
    model = timm.create_model(summary.get("backbone") or "vit_base_patch16_224",
                              pretrained=False, num_classes=num_classes)

    # Reconstruct the architecture: substitute each filter block (no pinv init needed
    # here -- the checkpoint's state_dict below overwrites all these weights anyway,
    # we just need matching shapes/parameter names), then inject LoRA + freeze.
    for idx in pruned_block_indices:
        substitute_filter_block(
            model, idx, num_layers=filter_block_layers,
            dropout=summary.get("filter_dropout") or 0.0,
            residual_hidden_dim=filter_residual_hidden_dim,
            residual_alpha=filter_residual_alpha, residual_dropout=filter_residual_dropout,
        )
    inject_lora(
        model, pruned_block_indices, lora_rank=lora_rank or 0,
        lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        adapter_type=adapter_type,
        paca_rank=paca_columns, paca_selection=summary.get("paca_selection") or "random",
        paca_tuner=summary.get("paca_tuner") or "direct",
        paca_adapter_rank=summary.get("paca_adapter_rank") or 0,
        unilora_dim=summary.get("unilora_dim") or 72000,
        unilora_seed=summary.get("seed") or 0,
    )
    # LoftQ is not rerun here: quantized weights and A/B already live in the checkpoint.
    freeze_non_trainable(model, pruned_block_indices)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, summary


def compute_lora_delta_report(model: torch.nn.Module):
    """Effective adaptation delta versus the stored base projection.

    For RPaCA this covers the current column interval only: earlier committed
    updates are already inside the saved base. It cannot reconstruct total
    change from original pretraining weights without those original weights.
    """
    rows = []
    for name, module in model.named_modules():
        with torch.no_grad():
            if isinstance(module, LoRALinear) and module.rank > 0:
                base = module.base_layer.weight
                delta = module.scaling * (module.lora_B @ module.lora_A)
                if module.adapter_type == "dora":
                    v = base + delta
                    delta = module.magnitude[:, None] * v / v.norm(dim=1, keepdim=True).clamp_min(1e-8) - base
            elif isinstance(module, PaCALinear):
                base = module.frozen_weight
                delta = module.paca_weight - base[:, module.selected_idx]
            elif isinstance(module, PaCAAdapterLinear):
                base = module.frozen_weight
                delta, _ = module._delta_subweight()
            elif isinstance(module, UniLoRAAdapterLinear):
                base = module.base_layer.weight
                a, b = module._reconstruct_AB()
                delta = module.scaling * (a @ b).T
                if module.adapter_type == "unidora":
                    v = base + delta
                    delta = module.lora_magnitude[:, None] * v / v.norm(dim=1, keepdim=True).clamp_min(1e-8) - base
            else:
                continue
            delta_norm = delta.norm().item()
            base_norm = base.norm().item()
        rows.append({"layer": name, "delta_norm": delta_norm, "base_norm": base_norm,
                     "ratio_pct": 100.0 * delta_norm / base_norm if base_norm else float("nan")})
    return rows


def print_report(rows: list):
    if not rows:
        print("[Diagnostic] No adaptation modules found in this model.")
        return

    print(f"\n{'Layer':<45} {'||delta||':>12} {'||base||':>12} {'delta/base %':>14}")
    print("-" * 85)
    for r in sorted(rows, key=lambda r: -r["ratio_pct"]):
        print(f"{r['layer']:<45} {r['delta_norm']:>12.4f} {r['base_norm']:>12.4f} {r['ratio_pct']:>13.2f}%")

    ratios = [r["ratio_pct"] for r in rows]
    mean_ratio = sum(ratios) / len(ratios)
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    print("-" * 85)
    print(f"{'MEAN':<45} {'':>12} {'':>12} {mean_ratio:>13.2f}%")
    print(f"Min: {min_ratio:.2f}% | Max: {max_ratio:.2f}% | N layers: {len(rows)}")

    print("[Diagnostic] Delta magnitude is descriptive, not proof of undertraining "
          "or useful adaptation. RPaCA deltas exclude previous committed epochs.")


def main():
    parser = argparse.ArgumentParser(
        description="Measure how much each LoRA adapter actually learned (delta magnitude vs base weight norm)."
    )
    parser.add_argument("run_output_dir", type=str,
                         help="Output dir of a --lora-rank > 0 training run (must contain "
                              "metrics_summary.json and the saved checkpoint).")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model, summary = load_run_model(args.run_output_dir, device=args.device)
    print(f"[Diagnostic] Loaded checkpoint for dataset={summary.get('dataset')}, "
          f"best_val_acc={summary.get('best_val_acc')}, final_test_acc={summary.get('final_test_acc')}")

    rows = compute_lora_delta_report(model)
    print_report(rows)


if __name__ == "__main__":
    main()