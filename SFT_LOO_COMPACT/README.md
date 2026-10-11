# LOO compact replacement and targeted PEFT

This is a separate experiment on branch `sft-loo-compact`. The four previous
backend folders and their GitHub branches are unchanged. Dataset split logic,
symbolic-filter math and PEFT implementations are copied without modification
from `sft-snip-compact` commit `214e43f`.

## Procedure

1. Load pretrained ViT-B/16 and the target-task classifier.
2. Train **only that classifier** for five epochs on the training split. Freeze
   the feature extractor, including LayerNorms, during this shared warm-up.
   Retain the warmed-up classifier for screening and final training. This is an
   explicit extension to make loss rankings meaningful with a new task head.
3. Cache up to `max(64, 4 * loo_samples)` training images, without advancing
   the training loader's sampling state. Choose up to `loo_samples` images by
   seeded round-robin sampling across observed labels in this pool. Report
   actual class coverage; this cannot create classes absent from the pool.
4. Measure the intact model's mean classification loss. Independently remove
   each original block and measure loss on **the exact same cached images**:

   `delta_loss[i] = loss(original_without_block_i) - loss(intact_original)`.

   Use evaluation mode and no gradients. Restore the original architecture
   after screening, including on errors. No recovery training occurs per
   candidate. For 12 blocks there are 13 forward evaluations, not 12 complete
   training runs. The criterion uses the same `--label-smoothing` as training.
5. Replace the **minimum signed delta-loss** block with the symbolic filter.
   Negative deltas are valid; choose the most negative, not the smallest absolute
   delta. Fit the filter from 64 training images using the existing residual
   pseudoinverse/SVD and bounded expression search. The final model still has
   12 positions: 11 original blocks plus the replacement.
6. Attach PEFT to the **maximum delta-loss surviving original block** only.
   Lowest original index wins ties. If all scores tie and the maximum is the
   replacement, select the highest-ranked remaining original block.
7. Train the filter, the selected block's adapters/selected columns, all
   LayerNorms and the classifier. Other dense weights are frozen. A runtime
   scope check and saved parameter manifest enforce this. All LayerNorms are
   the exception retained from existing SFT training, not extra block PEFT.
8. Save the best validation-accuracy checkpoint, restore it and evaluate test.
   Validation/test data never participate in the LOO ranking. Use `--defer-test`
   when comparing configurations, and select using validation before test.

```text
Original pretrained ViT + warmed-up task head
                    |
         same cached training images
                    |
   intact loss + 12 independent removal losses
                    |
            signed loss increases
          /                         \
   minimum increase            maximum surviving increase
          |                         |
  symbolic replacement       PEFT in that original block
          \                         /
        train filter + targeted PEFT + LN + head
                    |
      best validation checkpoint -> test evaluation
```

This is a LOO heuristic, not a reproduction of the paper's SNIP selection.
Low deletion loss need not predict symbolic replacement tolerance; high deletion
loss need not identify the best PEFT target. Results must establish both claims.
LOO costs 13 inference passes over the scoring subset plus one classifier warm-up
and one filter fit. Warm-up and final recovery training budgets are reported
separately. Compare these costs and use consistent classifier preparation when
comparing selection strategies. No improvement on CIFAR-100 is promised.

## Run from the repository root

```bash
# Default: symbolic rank 16, LoRA rank 16, 5 head warm-up epochs,
# 256 scoring images, early stopping up to 200 recovery epochs, patience 10.
python3 run.py loo-compact --dataset pets --seed 18 --device cuda

# Fixed common recovery budget. Head warm-up is an additional 5 epochs.
python3 run.py loo-compact --dataset cifar100 --adapter-type dora \
  --filter-type symbolic --epochs 100 --loo-samples 512 --device cuda

# Fused PaCA + DoRA on the selected original block.
python3 run.py loo-compact --dataset flowers102 --adapter-type paca \
  --paca-tuner dora --paca-rank 32 --paca-adapter-rank 16 --device cuda

# Replacement-only ablation, with no original-block PEFT.
python3 run.py loo-compact --dataset pets --adapter-type none --device cuda

# Full CLI; existing dense/lowrank filters are optional comparisons.
python3 run.py loo-compact --help
```

Supported: LoRA, DoRA, PaCA, RPaCA, PaCA+LoRA/DoRA and RPaCA+LoRA/DoRA.
Standalone LoRA/DoRA support existing LoftQ initialization. Orthogonality flags
are supported for standalone/fused LoRA/DoRA, not direct PaCA. Uni methods and
full-backbone fine-tuning are excluded. RPaCA resampling stays within the chosen
block and clears optimizer state for parameters whose column meaning changed.

All existing dataset choices remain available. Default `--num-samples 1000`
uses the existing dataset-specific split logic; `--use-full-dataset` retains the
existing full-data option. Increasing `--loo-samples` improves potential class
coverage but increases screening and pool memory. The actual dataset can be
smaller than the requested pool; actual counts and missing classes are recorded.
Initialization requires at least 64 training images. Setting
`--head-warmup-epochs 0` disables warm-up and prints an untrained-head warning.
`--warmup-epochs` instead controls the **final training learning-rate warm-up**.

## Results and checkpoints

Root CLI defaults to `outputs/loo-compact/loo_compact_..._<config_hash>/`:

- `loo_block_scores.csv`: one row per original block, intact/removal loss,
  signed increase, accuracies, sample counts and both selection flags.
- `loo_selection.json`: classifier warm-up, calibration coverage/pool indices,
  rankings, selected blocks and screening/fitting timings.
- `metrics_summary.json`: best validation accuracy, final test accuracy/loss,
  complete configuration, scope manifest, parameter counts and efficiency.
- `best_model.pt`: best validation state, including symbolic operators/scale and
  PaCA column identities. Restoration does not repeat ranking/warm-up/fitting.
- `filter_equations.json`, training history CSV and plots.

`param_breakdown.trainable_pct` uses the **retained model** as denominator.
`trainable_pct_original` uses the original full model. `total_reduction_pct`
measures retained-model parameter reduction and is different from the trainable
fraction. Dense frozen weights still occupy memory and checkpoints.

```bash
python3 run.py collect-all --root outputs --out compiled_results/all_results.csv
python3 run.py evaluate-loo-compact \
  --summary outputs/loo-compact/RUN/metrics_summary.json --device cuda
```

## A100 array

Create the log directory **before submitting**, and pre-download the selected
dataset/backbone into the shared cache before launching concurrent jobs. The
existing dataset downloader is available through `run.py download-data --help`.

```bash
mkdir -p /export/home/achyut/Sarvesh/sflogs
cd /export/home/achyut/Sarvesh/SFT_BEST
sbatch slurm/run_loo_compact_a100.sh --dataset pets --epochs 100
# Run only LoRA:
sbatch --array=0 slurm/run_loo_compact_a100.sh --dataset pets --epochs -1
```

The array has eight method configurations, not a manually trained block sweep.
Each task independently performs the same deterministic LOO ranking. Defaults
use `sft_env`, A100 on node1, one GPU per task and at most four concurrent tasks.
Set `CODE_DIR`/`CONDA_ENV` to override code/environment locations. Slurm log
paths must be overridden with sbatch if the account's filesystem differs.

## Verification

Run `python3 tools/verify_workflows.py` for all backends in isolated processes,
or `cd SFT_LOO_COMPACT && python3 -m pytest -q tests` for this workflow.
CPU tests use small ViTs and synthetic images and do not establish GPU training
accuracy, dataset download availability or cluster scheduling success.
