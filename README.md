# Final SFT research repository

Branch **`sft_final_res`** combines the original SFT/PEFT, compact filters,
block subtraction and LOO experiments. The four backend folders are preserved
byte-for-byte from the previously verified code. The dedicated SNIP experiment
(`SFT_SNIP_COMPACT`), its original launcher, evaluator and old download bundles are excluded.
Original trainers retain their existing selection helpers; use `--pruned-block`
for an explicit replacement position. Existing source branches are unchanged.

Download [downloads/SFT-final-res.zip](downloads/SFT-final-res.zip), or use
GitHub's **Code → Download ZIP** with `sft_final_res` selected. Extract it and run
commands from the directory containing **`run.py`**.

For separate campaigns using your existing `SFT_BEST`, `SFT_FINAL`, `SFT_EXP`,
`SFT_LOO` and `SFT_COEFF_RED` cluster folders, see
[slurm/research_suite/README.md](slurm/research_suite/README.md). Its seven A100
arrays and master compiler create the requested result folders, CSV tables and
PNG/SVG/PDF figures. A launcher can call the **external existing** SNIP backend;
that backend's source remains excluded here. A standalone launcher download is
available as [downloads/SFT-research-slurm-suite.zip](downloads/SFT-research-slurm-suite.zip).

## Repository contents

| Folder | Included methods |
| --- | --- |
| `SFT_ORIGINAL` | Original dense SFP filter, SFT/PEFT, full fine-tuning reference |
| `SFT_EXPERIEMTN` | Dense, low-rank and bounded symbolic replacement filters; grids and block sweeps |
| `SFT_BLOCK_SUBTRACTION` | Highest-correlation adjacent-pair reduction; independent one-block removal and validation selection |
| `SFT_LOO_COMPACT` | LOO loss screening; symbolic replacement at minimum loss increase; targeted PEFT at maximum surviving loss increase |
| `slurm` | A100/B200 launchers, original all-method grid and all-dataset LOO array |
| `tools` | Saved-checkpoint evaluators, CSV/log collectors and isolated test runner |

`run.py` launches each backend in its own process so similarly named modules
cannot interfere. Paths supplied on the CLI resolve from the caller's directory.
All applicable backend flags are forwarded; unsupported flags fail rather than
being silently ignored. `python3 run.py WORKFLOW --help` lists the exact options.
`python3 run.py --dry-run WORKFLOW FLAGS` previews a command without training.

## Environment and datasets

Use the cluster's existing CUDA-compatible PyTorch/torchvision environment:

```bash
conda activate sft_env
python3 -m pip install -r requirements.txt
python3 run.py --help
```

Preserve your existing CUDA installation when installing missing requirements.
The default backbone is `vit_base_patch16_224.augreg_in21k`.

All 13 existing dataset choices are available:
`pets`, `svhn`, `flowers102`, `dtd`, `caltech101`, `cifar100`, `fgvc_aircraft`,
`eurosat`, `sun397`, `pcam`, `clevr`, `dsprites-loc`, `dsprites-ori`.
Dataset loaders, proxies, augmentations and split rules are unchanged. Default
`--num-samples 1000` uses the existing 80/20 training/validation subset where
available; this is not full-dataset training. `--use-full-dataset` uses the
existing full-data split policy. Test splits remain as implemented in each loader.

Root CLI trainers share `data/`. Use `--data-dir /absolute/path/to/cache` to reuse
an existing cache. Prepare datasets/backbone before concurrent jobs. Download
availability depends on upstream hosts; the repository merge does not fix a
previously failing Caltech-101 annotation download URL.

The existing `download-data` helper supports eight datasets (see `--help`):

```bash
python3 run.py download-data --datasets cifar100 flowers102 svhn
```

It does not accept `pets`, `pcam`, `clevr` or the dSprites tasks; those are loaded
through their existing training dataset loaders. Cached datasets and pretrained
weights are not included in the source ZIP.

## Run each method

These examples use Pets, seed 18 and fixed 100 classification/recovery epochs.
Change `--dataset` to any supported choice. Correlation feature matching and
LOO classifier warm-up have additional, separately recorded budgets.

```bash
# 1. Original SFT: replace original block 3 with the dense SFP filter.
python3 run.py original --dataset pets --mode sft --pruned-block 3 \
  --seed 18 --epochs 100 --device cuda

# 2. Original SFT + LoRA, or use --adapter-type dora.
python3 run.py original --dataset pets --adapter-type lora --lora-rank 16 \
  --pruned-block 3 --seed 18 --epochs 100 --device cuda

# 3. Low-rank replacement, without adapters by default.
python3 run.py lowrank --dataset pets --filter-rank 16 --pruned-block 3 \
  --seed 18 --epochs 100 --device cuda

# 4. Compact symbolic replacement, without adapters by default.
python3 run.py symbolic --dataset pets --filter-rank 16 --pruned-block 3 \
  --seed 18 --epochs 100 --device cuda

# 5. Symbolic replacement plus DoRA in surviving original blocks.
python3 run.py symbolic --dataset pets --adapter-type dora --lora-rank 16 \
  --filter-rank 16 --pruned-block 3 --seed 18 --epochs 100 --device cuda

# 6. Highest-correlation adjacent pair: delete the later block,
#    recover the earlier block with LoRA. No 0.9 cutoff.
python3 run.py correlation --dataset pets --pair-index -1 --min-correlation -1 \
  --adapter-scope retained --adapter-type lora --lora-rank 16 \
  --calibration-samples 64 --distill-epochs 5 \
  --seed 18 --epochs 100 --device cuda

# 7. Physically delete block 3 and apply DoRA to survivors.
python3 run.py drop-one --dataset pets --remove-block 3 --adapter-type dora \
  --lora-rank 16 --seed 18 --epochs 100 --device cuda

# 8. LOO: minimum signed removal-loss increase -> symbolic replacement;
#    maximum surviving increase -> PEFT in that original block only.
python3 run.py loo-compact --dataset pets --filter-type symbolic --filter-rank 16 \
  --adapter-type lora --lora-rank 16 --loo-samples 512 \
  --head-warmup-epochs 5 --seed 18 --epochs 100 --device cuda

# 9. LOO replacement-only control, with no original-block PEFT.
python3 run.py loo-compact --dataset pets --adapter-type none \
  --loo-samples 512 --head-warmup-epochs 5 \
  --seed 18 --epochs 100 --device cuda

# 10. Full fine-tuning reference.
python3 run.py original --dataset pets --mode full_finetune \
  --seed 18 --epochs 100 --device cuda
```

For early stopping, replace `--epochs 100` with:

```bash
--epochs -1 --max-epochs 200 --patience 10
```

Both protocols use the existing cosine schedule and restore the best validation
checkpoint before test evaluation. Cosine scheduling and early stopping are
separate choices; they can operate together.

## PEFT variants and placement

Original, compact-filter and subtraction trainers support LoRA, DoRA, PaCA,
RPaCA, Uni-LoRA and Uni-DoRA, plus applicable fused PaCA/RPaCA tuners,
LoftQ initialization and orthogonality flags. LOO excludes the Uni methods and
supports `none` for its replacement-only control.

For any compatible training workflow, substitute these adapter flags:

| Variant | Flags |
| --- | --- |
| LoRA | `--adapter-type lora --lora-rank 16` |
| DoRA | `--adapter-type dora --lora-rank 16` |
| PaCA | `--adapter-type paca --lora-rank 32 --paca-tuner direct` |
| RPaCA | `--adapter-type rpaca --lora-rank 32 --paca-tuner direct` |
| PaCA + LoRA | `--adapter-type paca --paca-tuner lora --paca-rank 32 --paca-adapter-rank 16` |
| PaCA + DoRA | `--adapter-type paca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 16` |
| RPaCA + LoRA | `--adapter-type rpaca --paca-tuner lora --paca-rank 32 --paca-adapter-rank 16` |
| RPaCA + DoRA | `--adapter-type rpaca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 16` |
| Uni-LoRA (outside LOO) | `--adapter-type unilora --lora-rank 16 --unilora-dim 72000` |
| Uni-DoRA (outside LOO) | `--adapter-type unidora --lora-rank 16 --unilora-dim 72000` |

For example:

```bash
python3 run.py loo-compact --dataset cifar100 --filter-type symbolic \
  --adapter-type paca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 16 \
  --loo-samples 512 --head-warmup-epochs 5 --seed 18 --epochs 100 --device cuda

python3 run.py original --dataset pets --adapter-type lora --lora-rank 16 \
  --init-method loftq --pruned-block 3 --seed 18 --epochs 100 --device cuda
```

Adapter placement differs by workflow. Original/ordinary compact training
adapts surviving blocks. Correlation defaults to only the retained earlier
block; `--adapter-scope all` is an explicit alternative. Physical removal adapts
survivors. LOO adapts only the maximum-loss surviving original block.
The existing task-head and LayerNorm training policies remain active. These
parameters are additional to PEFT and replacement-filter parameters.

LOO measures signed mean classification-loss increases on identical cached
training images after a shared classifier-only warm-up. It does not train 12
recovery candidates. For a 12-block model it screens the intact model plus 12
independent deletions, restores the original architecture, and replaces one
position. Its scoring uses no validation/test data. Minimum loss increase and
maximum surviving loss increase are heuristics, not guarantees of the best
replacement or adaptation position.

## Block sweeps and validation selection

```bash
# Manual all-position SFT sweep: 12 independent models.
for block in {0..11}; do
  python3 run.py original --dataset pets --mode sft --pruned-block "$block" \
    --seed 18 --epochs 100 --device cuda
done

# Full physical-removal sweep: 6 families x 12 blocks = 72 candidates.
# Each candidate defers test; the completed sweep selects with validation
# and then evaluates the winning removal checkpoint for each configuration.
python3 run.py removal-sweep --dataset pets --seeds 18 --epochs 100 --device cuda

# Prepare those candidates for a Slurm array without training/submitting.
python3 run.py removal-sweep --prepare-only --dataset pets --seeds 18 \
  --epochs 100 --device cuda
```

The printed removal manifest supplies the root and task count. For the default
72 tasks, submit:

```bash
sbatch slurm/run_integrated_removal_array_a100.sh /absolute/path/manifest.json
# After all tasks complete; use the manifest's parent folder:
python3 run.py collect-removals --root /absolute/path/removal_sweep_folder \
  --seeds 18 --evaluate-selected --device cuda
```

If method/seed/block counts change, override the array range. Incomplete
candidate coverage is rejected; `--allow-partial` explicitly limits selection
to available candidates. It does not establish the best of all 12.

Existing original and dense/low-rank/symbolic grids are also available:

```bash
python3 run.py original-grid prepare --training-budget both --seeds 18
python3 run.py original-grid task --manifest /absolute/path/manifest.json --task-id 0
python3 run.py original-grid collect --manifest /absolute/path/manifest.json

python3 run.py symbolic-grid prepare --datasets pets cifar100 \
  --block-counts 1 2 4 --ranks 16 --training-budget both
python3 run.py symbolic-grid run --manifest /absolute/path/manifest.json --task-id 0
```

The compact all-position submitter is preserved; override its old source path:

```bash
CODE_DIR="$PWD/SFT_EXPERIEMTN" bash SFT_EXPERIEMTN/submit_block_sweep.sh \
  --datasets pets cifar100 --ranks 16 --training-budget both
```

## Slurm launchers

Submit from the extracted repository root (the directory containing `run.py`),
or export `CODE_DIR` for the integrated launchers. Create the log directory
before `sbatch`. A100 jobs use your existing partition, node1, one GPU, 220G and
11:59:59. Conda defaults to `sft_env`.

```bash
mkdir -p /export/home/achyut/Sarvesh/sflogs

# Any single workflow.
sbatch slurm/run_integrated_a100.sh symbolic --dataset pets --pruned-block 3 --epochs 100 --device cuda

# Highest-correlation reduction, Pets + LoRA by default.
sbatch slurm/run_correlation_a100.sh --epochs 100

# LOO: 8 PEFT configurations on Pets (override --dataset if needed).
sbatch slurm/run_loo_compact_a100.sh --epochs 100

# LOO: all 13 datasets x 8 PEFT configurations = 104 tasks, at most 4 concurrent.
sbatch slurm/run_loo_all_datasets_a100.sh --epochs 100

# Original: 13 datasets x 20 named configurations = 260 tasks per seed/budget.
# The helper prepares its manifest/log directories before submitting.
bash slurm/submit_all.sh --training-budget both --dry-run
bash slurm/submit_all.sh --training-budget both
```

LOO includes LoRA, DoRA, PaCA, RPaCA and the four fused variants; it excludes
Uni methods. Array defaults use early stopping; `--epochs 100` selects a fixed
recovery budget. Default LOO head warm-up is an additional five epochs. Each
array task independently repeats deterministic LOO selection and writes its own
run folder. Prepare shared caches before concurrent LOO tasks; this launcher
does not add download locking to the underlying trainer.

The original grid uses its existing shared dataset/backbone download locks and
automatic aggregation. The submission helper now defaults to `SFT_ORIGINAL`
and records the root helper path separately from the trainer path.

For B200, create `/home/achyutm01/SFT/sflogs` and submit
`slurm/run_integrated_b200.sh WORKFLOW FLAGS` from the root. It retains your
`mig90g`, `gpu:3g.90gb:1`, 64G settings. Set `CONDA_ENV` or override sbatch output
paths when your environment/account differs.

## Results, plotting and compilation

Root CLI trainers default to `outputs/WORKFLOW/`; array/grid launchers use their
own run roots beneath `outputs/`. Each backend keeps its existing summaries,
training history, plots and best-checkpoint format. LOO additionally writes
`loo_block_scores.csv`, `loo_selection.json` and `filter_equations.json`.
Correlation writes `adjacent_correlations.csv/.json` and distillation history.

```bash
# Compile every saved summary into one CSV; no retraining or test evaluation.
python3 run.py collect-all
# -> compiled_results/all_results.csv

# Original accuracy/parameter analyses and available plots.
python3 run.py collect-original --root outputs/original --seeds 18

# Compact-filter comparisons.
python3 run.py collect-symbolic --root outputs --out compiled_results/compact_results.csv

# Evaluate saved LOO or subtraction checkpoints without retraining.
python3 run.py evaluate-loo-compact --summary /absolute/path/metrics_summary.json --device cuda
python3 run.py evaluate-removal --summary /absolute/path/metrics_summary.json --device cuda

# Compile existing epoch logs.
python3 run.py collect-logs --root /absolute/path/log_directory \
  --jobs out:27703 out1:27716 out2:27718
```

The general CSV preserves null/deferred test results, selected indices and full
saved configurations. `collect-symbolic` uses the existing broad experiment
collector; use `collect-all` when combining all workflow types. Use validation
for selecting hyperparameters/blocks and compare test results under matched
settings. Trainable parameter fraction, total retained parameters and checkpoint
storage are different quantities; frozen backbone weights still occupy memory.

## Verification and provenance

```bash
python3 -m pip install -r requirements-dev.txt
python3 tools/verify_workflows.py
```

The verifier runs each backend in isolation. See
[RESEARCH_VERIFICATION.md](RESEARCH_VERIFICATION.md) for executed checks and
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) for preserved source references.
Existing per-backend reports describe their earlier verification stages.
CPU checks do not establish GPU accuracy, cluster scheduling success or exact
paper reproduction. Final research claims require your matched dataset runs.
