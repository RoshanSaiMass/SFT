# LOO-selected GELU bottleneck with targeted PEFT

This separate backend preserves the existing symbolic and SFP implementations.

1. Warm up only the task head for five epochs on training images.
2. Independently bypass each original ViT block on a shared, training-only,
   class-balanced calibration subset. Choose the minimum signed CE loss increase
   for replacement and the maximum surviving increase for PEFT.
3. Replace the selected block with
   `x + up(GELU(down(LayerNorm(x))))`.
4. Initialize down-projection using PCA and up-projection using least squares.
   Run 200 local reconstruction steps on original-block training activations.
   Select the best reconstruction state, including step zero, using 16 training
   images held out from the 48-image initialization fit. Token budgets cap each
   partition at 2048. No validation/test data enter fitting or LOO selection.
5. Train the bottleneck, PEFT in one surviving original block, all LayerNorms,
   and classifier with classification loss; other pretrained weights stay frozen.
6. Early stopping selects the best validation checkpoint, then evaluates test.

This is a fixed GELU nonlinear bottleneck, not symbolic expression selection.
There is no attention/token mixing inside the replacement and no full-network
teacher distillation during task training. Better accuracy than SFP is unproven.
Local imitation adds a training stage; report its time as well as task training.
The head-only warmup and backbone used during screening are the same as the
existing LOO method, not a claim to reproduce SFP's SNIP selection.

## Run

Extract this branch with `run.py`, `tools/`, and `SFT_LOO_BOTTLENECK/` directly
under `/export/home/achyut/Sarvesh/SFT_BOTTLENECK` (or set `--repo-root`).
The independent scheduler needs no `research_common.sh`:

```bash
mkdir -p /export/home/achyut/Sarvesh/sflogs
sbatch /export/home/achyut/Sarvesh/SFT_BOTTLENECK/scheduler/loo_bottleneck.slurm \
  --campaign bottleneck64_lora8 --filter-rank 64 --peft-rank 8 --adapter-type lora
```

The array runs the same 12 datasets as the num_function grid, one dataset per task.
For one Pets smoke run, add `sbatch --array=0` before the script path. For DoRA use
`--adapter-type dora` and a different campaign. Widths 64, 96, 128 and PEFT ranks
8, 16 are configurable; positive fixed `--epochs 100` overrides early stopping.
Supported adapters: none, lora, dora, paca, rpaca; fused variants use
`--adapter-type paca --paca-tuner lora` (or dora/rpaca) with `--paca-rank 32`.
For a fused variant PEFT rank must not exceed PaCA columns.

Direct CLI:

```bash
python run.py loo-bottleneck --dataset pets --device cuda \
  --filter-rank 64 --adapter-type lora --lora-rank 8 --lora-alpha 16 \
  --loo-samples 512 --head-warmup-epochs 5 --epochs -1 \
  --data-dir /path/to/data --output-dir /path/to/results
```

Outputs include `loo_selection.json`, `loo_block_scores.csv`,
`initial_filter_equations.json`, `filter_equations.json` (architecture and
reconstruction reports), training history/plots, `metrics_summary.json` with
validation/test metrics and measured parameter counts, and `best_model.pt`.
The default checkpoint stores trained parameters and required buffers; retain
its summary and pretrained backbone cache to restore. Full state is optional
with `--checkpoint-format full`. Verified atomic saves detect storage failures;
free disk space is still required.

```bash
python run.py evaluate-loo-bottleneck --summary /path/to/metrics_summary.json --device cuda
python tools/count_bottleneck.py --num-classes 37
```

## Measured parameter budget (Pets, 37 classes)

Denominator: original task-head ViT-B/16, 85,827,109 parameters. Counts include
filter, one-block PEFT, every trainable LayerNorm and head; no double counting.
Other datasets change classifier count by 769 parameters per added class.
These percentages are trainable fractions of the original model, not the
fraction of all weights retained for inference. Roughly 79 million parameters
remain in the modified model, depending on configuration.

| Width | PEFT rank | LoRA total | LoRA % | DoRA total | DoRA % |
|---|---|---|---|---|---|
| 64 | 8 | 262,757 | 0.3061 | 269,669 | 0.3142 |
| 64 | 16 | 361,061 | 0.4207 | 367,973 | 0.4287 |
| 96 | 8 | 311,941 | 0.3635 | 318,853 | 0.3715 |
| 96 | 16 | 410,245 | 0.4780 | 417,157 | 0.4860 |
| 128 | 8 | 361,125 | 0.4208 | 368,037 | 0.4288 |
| 128 | 16 | 459,429 | 0.5353 | 466,341 | 0.5433 |

Analytic total for LoRA on this backbone: `1537*width + 12288*rank + 37632 + 769*classes`.
DoRA adds 6912 magnitude parameters. Filter alone: `1537*width + 2304`.
The count tool instantiates the actual architecture without pretrained downloads
and verifies training scope. Replacing any of the 12 uniform blocks gives the
same budget. A lower trainable count does not establish better accuracy.

## Validation

CPU tests exercise training-only reconstruction, gradients, all nine supported
adapter/tuner cases, actual short end-to-end LOO training, frozen-weight scope,
saved checkpoint reconstruction, repeated test evaluation, and failed writes.
No GPU benchmark or dataset accuracy improvement is claimed.
