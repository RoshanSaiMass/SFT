# Physical block subtraction using the original SFT/PEFT code

This folder is independent of `SFT_EXPERIEMTN`. It was created on branch
`sft-block-subtraction` from original `main` commit `0245af6`, using
`SFT_LoRA_2-reviewed.zip` (SHA256
`c82fdc921eb31f03e1e79c72308c29efe6be85fa92249fea0294653662b56901`).
The original PEFT implementation `single_filter_lora.py`, original trainer
`train_sfp_lora.py`, and original tests are copied unchanged. The new entry point
is `train_subtraction.py`; it deletes a transformer block physically. It does not
insert an SFP filter, a symbolic filter, or a pseudoinverse approximation.

## 1. Correlation-selected adjacent pair

```text
Original:   ... -> B_i -> B_(i+1) -> remaining blocks -> classifier
                       training images supply y_i and y_(i+1)
Selection:  mean per-image Pearson(y_i, y_(i+1)) across all tokens/channels
Teacher:    cache x before B_i and target B_(i+1)(B_i(x))
Student:    ... -> B_i + LoRA -> remaining blocks -> classifier
                      ^ B_(i+1) has been deleted
Training:   feature matching on cached training examples, then classification
```

Measure all 11 adjacent pairs of the pretrained 12-block ViT on 64 training
images by default. Select the highest mean Pearson correlation meeting 0.9.
Delete the later block and inject LoRA into the retained earlier block's QKV,
attention projection and two MLP linear layers. The teacher feature targets
are cached before deletion; no second full model is kept during optimization.
Five feature-matching epochs learn the two-block output before classification
fine-tuning. The feature loss is normalized MSE; optional existing orthogonality
regularization also applies. Correlation, cosine similarity and output errors
are saved to `adjacent_correlations.csv` and `.json`.

High correlation is a heuristic: scale or offset changes can retain Pearson 1
while outputs differ. Distillation attempts recovery; it does not guarantee
equivalence or accuracy. If no pair qualifies, the run saves its scores and
stops, rather than deleting an unqualified block. `--pair-index 3` requests
original blocks 3/4 but still checks the threshold. Use `--min-correlation` to
change the threshold explicitly. Indices are zero-based.

```bash
python3 train_subtraction.py --method correlation --dataset pets \
  --adapter-type lora --lora-rank 16 --seed 18 --device cuda \
  --calibration-samples 64 --min-correlation 0.9 --distill-epochs 5
```

Defaults put adapters only in the retained block. The original training policy
also trains LayerNorms and the classifier. `--train-layernorms target` limits
LayerNorm training to adapted blocks and final norm; `none` freezes LayerNorms.
`--adapter-scope all` adapts all survivors as a separate experimental setting.
`--distill-epochs 0` disables feature matching for a useful control experiment.
Other supported PEFT families can replace LoRA explicitly.

## 2. Exhaustive one-block removal with PEFT recovery

For every candidate, start from the same pretrained model, delete exactly one
original block, then fit PEFT adapters in the 11 survivors. This is an independent
12-candidate sweep, not cumulative deletion. Choose the best removal by
validation accuracy; evaluate the chosen checkpoint on test afterward.

```bash
# One candidate, without inspecting test during selection:
python3 train_subtraction.py --method drop-one --remove-block 3 \
  --dataset pets --adapter-type dora --lora-rank 16 --seed 18 \
  --device cuda --defer-test

# Sequential sweep: 6 families x 12 blocks = 72 runs per seed.
python3 run_removal_sweep.py --dataset pets --seeds 18 --device cuda

# Just LoRA / DoRA / PaCA, three seeds = 108 runs:
python3 run_removal_sweep.py --dataset cifar100 --adapter-types lora dora paca \
  --seeds 18 19 20 --epochs 100 --device cuda
```

Supported families are `lora`, `dora`, `paca`, `rpaca`, `unilora`, `unidora`.
Original fused PaCA/RPaCA with `--paca-tuner lora` or `dora`, LoftQ initialization
for standalone LoRA/DoRA, and supported orthogonality penalties remain available.
For example, use `--adapter-types paca rpaca --paca-tuner lora --paca-rank 32
--paca-adapter-rank 16` for a fused sweep. Incompatible options fail validation
before training. PaCA direct columns and RPaCA resampling follow the original
implementation. LoRA/DoRA dense base weights remain frozen; DoRA magnitudes
are trainable. The head and LayerNorm policy above still apply.

The default is early stopping (`--epochs -1 --max-epochs 200 --patience 10`).
`--epochs 100` runs a fixed 100 epochs. Both use cosine scheduling and restore
the best validation checkpoint. Feature-matching epochs are additional to the
classification budget; log and report this cost when comparing methods.
The original optimizer groups, learning rates, label smoothing 0.1 and split
rules are retained. Selection uses mean best validation accuracy across requested
seeds and chooses a **single common block per matched configuration**. Different
ranks, tuners, datasets and budgets are grouped separately. Ties favor the smaller
block index. Test accuracy never ranks candidates.

## Cluster arrays

Create SLURM's log directory before submitting: SLURM opens logs before the
script runs. Activate your environment, change into this folder, and prepare a
manifest on the login node. **Pass `--device cuda` during preparation**, because
the manifest stores the worker device even when the login node has no GPU.

```bash
mkdir -p /export/home/achyut/Sarvesh/logs
python3 download_data.py --datasets pets --data-root ./data
python3 run_removal_sweep.py --prepare-only --device cuda --dataset pets \
  --adapter-types lora dora paca rpaca unilora unidora --seeds 18 --epochs 100
# Copy the printed absolute Manifest path below (72 tasks => IDs 0..71):
sbatch slurm/removal_a100.sh /absolute/path/to/manifest.json
```

A100 defaults: partition `gpu-a100`, node `node1`, one GPU, 220G host RAM, four
concurrent tasks. B200 defaults: partition `mig90g`, one `gpu:3g.90gb:1` slice,
64G host RAM, two concurrent tasks. For B200 create `/home/achyutm01/SFT/logs`
and use `slurm/removal_b200.sh`. The worker directory defaults to
`/export/home/achyut/Sarvesh/SFT_EXP/SFT_BLOCK_SUBTRACTION` on A100 or
`/home/achyutm01/SFT/SFT_BLOCK_SUBTRACTION` on B200; override with `CODE_DIR`.
Both default to Conda `sft_env`; override with `CONDA_ENV`.
Download the chosen dataset once before launching concurrent tasks, or pass
`--data-dir` pointing to an already complete cache. This avoids simultaneous
first-use downloads into the same directory.

Change the submitted array range to match the manifest's printed task count:
`sbatch --array=0-11%4 slurm/removal_a100.sh /path/manifest.json` for only LoRA,
or `0-215` for six families, three seeds. Each task runs one candidate, records
its status and refuses CPU fallback if CUDA is missing. `--resume` skips only
an already completed matching run with a checkpoint. Failed tasks return an
error to SLURM; unfinished candidates prevent complete selection.

For a single correlation job:

```bash
sbatch slurm/correlation_a100.sh --epochs 100
```

## Results, selection and datasets

Each run has its own folder under `outputs/` with `metrics_summary.json`,
`best_model.pt`, epoch CSV, training/LR/parameter plots and (for correlation)
the correlation table and distillation history. Summaries record original block
indices, current-to-original mappings, adapter locations, trainable and total
parameter counts, real deletion size, checkpoint bytes, timing and GPU memory.
Trainable fraction and total retained model size are distinct: deleting one of
12 blocks does **not** reduce the complete model to the adapter size. With
adapters on all survivors, additional parameters partly offset deletion savings.

Sequential sweeps automatically collect results. After a SLURM array finishes,
run the following on a GPU allocation (test evaluation uses CUDA):

```bash
python3 select_removals.py --root /absolute/path/to/removal_sweep_TIMESTAMP \
  --seeds 18 --evaluate-selected --device cuda
```

This writes `compiled_results/results_all.csv`, `best_removals.csv` and
`selection.json`. Supply every planned seed with `--seeds` so entirely missing
seeds cannot be silently omitted when collecting standalone runs. For a manifest
sweep, planned seeds, blocks and method configurations are checked automatically,
including methods with no completed runs. Full block coverage 0..11 is required
by default without a manifest. If deliberately sweeping a subset, specify it with `--blocks 0 1 2`.
Incomplete sweeps still yield the candidates CSV but refuse winner evaluation.
Candidates retain null test accuracy; only selected checkpoints get tested.
For correlation-only outputs the collector simply combines the summaries.

All original dataset choices remain supported; examples include `pets`,
`cifar100`, `flowers102`, `caltech101`, `dtd`, `fgvc_aircraft` and `eurosat`.
Set `--data-dir /path/to/existing/data` to reuse downloads. The copied loader
preserves splits and adds RGB conversion, correct Caltech101 metadata through
Subset wrappers, and a configurable cache directory. Download dependencies are
included. Use the same sample-count/full-dataset flags and splits as your SFT
baseline; default `--num-samples 1000` is **not** full-dataset training.

## Development

```bash
python3 -m pip install -r requirements.txt -r requirements-dev.txt
python3 -m pytest -q tests
```

CPU tests exercise deletion, pair selection, feature matching, PEFT combinations,
full checkpoint restoration, deferred test evaluation, complete validation-based
selection and manifest execution. They do not establish full-dataset accuracy or
cluster resource availability. Existing `AUDIT_REPORT.md` and `PEFT_AUDIT.md`
describe the copied original implementation, not claims about these new methods.
