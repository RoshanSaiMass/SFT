# Integrated SFT, PEFT, compact filters and block subtraction

Branch `sft-integrated` combines these sources without changing their original
branches:

| Source branch | Source commit | Included code |
| --- | --- | --- |
| `main` | `0245af6` | Original reviewed SFT/PEFT, materialized as `SFT_ORIGINAL` |
| `sft-experiment` | `9cc4fc0` | `SFT_EXPERIEMTN`: dense, low-rank and symbolic replacement |
| `sft-block-subtraction` | `49ecab8` | `SFT_BLOCK_SUBTRACTION`: correlation merging and removal sweep |

Use **`run.py` from this repository's top directory**. It launches the appropriate
trainer in a separate Python process. Original CLI arguments are forwarded,
including options written as `--flag=value`. Identically named modules remain
isolated. Relative data, output, manifest and summary paths resolve from the
directory where you call the CLI. Existing direct entry points remain available.

The integrated original copy adds the same RGB/Caltech metadata/cache fixes
already checked in the other workflows and a `--data-dir` CLI flag. Dataset split
rules and the PEFT implementation are preserved. Original run names use the
compact branch's existing readable-prefix/hash handling when lengthy CLI options
would exceed filesystem limits. The two experimental folders
are merged unchanged from their source branches.

## Install and choose a workflow

Use your existing CUDA environment, then install missing dependencies:

```bash
conda activate sft_env
python3 -m pip install -r requirements.txt
python3 run.py --help
python3 run.py original --help
python3 run.py symbolic --help
python3 run.py correlation --help
python3 run.py removal-sweep --help
```

`WORKFLOW --help` prints every flag supported by that backend. Options belong to
their applicable workflow: a symbolic search flag cannot be applied to physical
deletion, for example. The backend rejects unsupported or incompatible settings.
Use `python3 run.py --dry-run WORKFLOW FLAGS` to inspect the exact command without
training, dataset downloads or model downloads.

| Workflow | What it does |
| --- | --- |
| `original` / `dense` | Original SFP filter, original PEFT variants, full fine-tuning |
| `lowrank` | Compact low-rank replacement filter |
| `symbolic` | Bounded symbolic residual replacement filter |
| `correlation` | Delete the later of a correlated adjacent pair and adapt the earlier block |
| `drop-one` | Delete a specified original block and adapt survivors |
| `removal-sweep` | Independent deletion candidates across blocks, seeds and PEFT families |
| `original-grid` | Existing original all-dataset/all-configuration manifest launcher |
| `symbolic-grid` | Existing dense/low-rank/symbolic compression grid |
| `collect-original` | Original analysis, CSVs and plots with original options |
| `collect-symbolic` | Compact experiment summary collector |
| `collect-removals` | Validation-selected removal results and optional test evaluation |
| `collect-all` | One CSV of every saved summary, preserving configurations |
| `collect-logs` | Compile epoch logs, including the supplied `out` / `out1` / `out2` format |
| `evaluate-removal` | Test one saved subtraction checkpoint without retraining |
| `download-data` | Download selected datasets into the shared cache |

## Training examples

```bash
# Original SFT baseline, manual replacement of block 3.
python3 run.py original --dataset pets --mode sft --pruned-block 3 --seed 18 --device cuda

# Original SFT + DoRA. All six PEFT families are available in applicable trainers.
python3 run.py original --dataset pets --adapter-type dora --lora-rank 16 --pruned-block 3 --seed 18 --device cuda

# Fused PaCA/DoRA with orthogonality penalties.
python3 run.py original --dataset pets --adapter-type paca --paca-tuner dora \
  --paca-rank 32 --paca-adapter-rank 16 --lora-ortho-lambda1 0.9 \
  --lora-ortho-lambda2 0.9 --pruned-block 3 --seed 18 --device cuda

# Original LoRA with LoftQ initialization.
python3 run.py original --dataset pets --adapter-type lora --lora-rank 16 --init-method loftq --device cuda

# Full fine-tuning reference.
python3 run.py original --dataset cifar100 --mode full_finetune --epochs 100 --device cuda

# Low-rank and symbolic replacements; without PEFT flags, default to mode sft.
python3 run.py lowrank --dataset pets --filter-rank 16 --pruned-block 3 --seed 18 --device cuda
python3 run.py symbolic --dataset pets --filter-rank 16 --pruned-block 3 --seed 18 --device cuda

# Symbolic replacement with DoRA in the untouched blocks.
python3 run.py symbolic --dataset pets --adapter-type dora --lora-rank 16 --pruned-block 3 --device cuda

# More than one symbolic replacement; keep at least one original attention block.
python3 run.py symbolic --dataset cifar100 --filter-rank 16 --num-filter-blocks 4 --epochs 100 --device cuda

# Highest-valid-correlation pair, then 5 feature matching epochs + classification.
python3 run.py correlation --dataset pets --adapter-type lora --lora-rank 16 --seed 18 --device cuda

# Optional strict correlation cutoff (refuses deletion if no pair qualifies).
python3 run.py correlation --dataset pets --min-correlation 0.9 --device cuda

# Independently delete original block 3, then DoRA recovery; defer candidate testing.
python3 run.py drop-one --dataset pets --remove-block 3 --adapter-type dora --lora-rank 16 --defer-test --device cuda

# Six families x 12 candidates = 72 runs per seed, sequentially on one GPU.
python3 run.py removal-sweep --dataset pets --seeds 18 --epochs 100 --device cuda
```

PEFT options include LoRA, DoRA, PaCA, RPaCA, Uni-LoRA, Uni-DoRA, fused PaCA
tuners, supported LoftQ initialization and supported orthogonality regularization.
Existing filter depth, residual, manual/SNIP/ablation selection, adapter rank,
symbolic operator search, seed, sample budget, image export and plotting flags
remain available in the trainers that implement them. See each workflow's help
and copied README for exact constraints.

`--epochs -1 --max-epochs 200 --patience 10` uses early stopping; `--epochs 100`
uses fixed epochs. Best validation checkpoints are restored in both cases.
Correlation feature matching has a separate additional budget. The integrated
correlation default is `--min-correlation -1`, as requested: select the highest
valid pair rather than requiring 0.9. High correlation does not guarantee safe
removal or equal output behavior.

All 13 original dataset choices remain available. Default sample budget is
1,000 selected training-pool images; it is not full-dataset training. Preserve
your comparison budget and use `--use-full-dataset` explicitly if intended.

## Shared data and results

The common CLI defaults to `data/` beside `run.py` for every trainer and the data
downloader. Use `--data-dir /path/to/existing/data` to reuse your complete cache.
Download a dataset once before concurrent jobs to avoid first-use download races.

```bash
python3 run.py download-data --datasets pets cifar100
python3 run.py symbolic --dataset pets --data-dir /export/home/achyut/Sarvesh/SFT_FINAL/data --device cuda
```

Runs are placed under `outputs/original`, `outputs/lowrank`, `outputs/symbolic`,
`outputs/correlation`, `outputs/drop-one` and `outputs/removal-sweep`. Each backend
keeps its per-run naming, summaries, epoch history, plots and checkpoints.
`--output-dir` overrides this location. Direct backend entry points retain their
original defaults; use `--root` explicitly to collect legacy output trees.

```bash
# Every completed summary in one CSV. Does not select or evaluate anything.
python3 run.py collect-all
# Output: compiled_results/all_results.csv

# Original baseline analysis and its existing plot options.
python3 run.py collect-original --seeds 18 --no-plots

# Removal selection and test evaluation: run on an allocated GPU.
python3 run.py collect-removals --root outputs/removal-sweep/removal_sweep_TIMESTAMP \
  --seeds 18 --evaluate-selected --device cuda

# Evaluate one checkpoint even if a complete removal sweep is unavailable.
python3 run.py evaluate-removal --summary /absolute/path/metrics_summary.json --device cuda

# Compile existing log jobs into CSV. Pass the actual directory containing logs.
python3 run.py collect-logs --root /export/home/achyut/Sarvesh/SFT_EXP/SFT_EXPERIEMTN \
  --jobs out:27703 out1:27716 out2:27718
```

Removal candidates have null test accuracy when deferred. The removal selector
chooses a common block by mean validation accuracy across requested seeds and
checks complete candidate coverage. `--allow-partial` explicitly permits selection
among available candidates; that is not the best of all 12 if candidates are
missing. The general CSV collector preserves null values and does not silently
test or fill missing accuracies. Trainable parameter fraction, total retained
parameters and actual checkpoint storage are recorded as distinct quantities.

## A100 / B200 submission

Submit from the extracted repository directory containing `run.py`; alternatively
export `CODE_DIR` to that absolute directory. Conda defaults to `sft_env` and can
be overridden with `CONDA_ENV`.

```bash
mkdir -p /export/home/achyut/Sarvesh/sflogs
sbatch slurm/run_integrated_a100.sh symbolic --dataset pets --pruned-block 3 --filter-rank 16 --device cuda
sbatch slurm/run_integrated_a100.sh correlation --dataset pets --device cuda

# Prepare on the login node; store CUDA in the manifest explicitly.
python3 run.py removal-sweep --prepare-only --dataset pets --adapter-types lora dora paca rpaca unilora unidora --seeds 18 --epochs 100 --device cuda
# Copy the printed manifest path; 72 tasks => IDs 0..71.
sbatch slurm/run_integrated_removal_array_a100.sh /absolute/path/manifest.json
```

Override the array range if you change method, seed or candidate count. The
existing root `slurm/` original sweep and compact branch launchers remain included.
Their original hardcoded paths should be overridden when using a new directory;
use the new integrated launchers for individual workflows.

For B200, create `/home/achyutm01/SFT/sflogs` and use
`slurm/run_integrated_b200.sh`. It requests `mig90g` and `gpu:3g.90gb:1` with no
A100 node constraint. A100 integrated launchers request `gpu:1`, including when
you choose a compilation workflow: your cluster reported `QOSMinGRES` for a
job without the minimum GPU allocation.

Original and compact grid CLI examples:

```bash
python3 run.py original-grid prepare --training-budget both --seeds 18
python3 run.py original-grid task --manifest /path/manifest.json --task-id 0
python3 run.py original-grid collect --manifest /path/manifest.json

python3 run.py symbolic-grid prepare --datasets pets cifar100 --block-counts 1 2 4 --ranks 16 --training-budget both
python3 run.py symbolic-grid run --manifest /path/manifest.json --task-id 0
```

The original grid prepares all original datasets/configurations; inspect the
printed manifest/task count before submitting an array. The compact grid tests
replacement counts, while the included compact block-sweep scripts test each
individual position. No jobs are submitted merely by preparing a manifest.

## Verification and download

```bash
python3 -m pip install -r requirements-dev.txt
python3 tools/verify_workflows.py
```

The verifier runs root integration tests and each backend's tests in separate
processes. Do not collect all backend tests in a single pytest process: they
intentionally retain the same original module names. See `INTEGRATION_VERIFICATION.md`
for executed checks and limitations. Existing branch audit reports describe their
own source implementations; competitive accuracy and paper equivalence remain
experimental claims requiring dataset runs.

Download this branch through GitHub's Code / Download ZIP, or use the committed
`downloads/SFT-integrated.zip` source bundle. Both include the common CLI, all
three source folders, requirements and launchers; neither includes datasets,
pretrained weights or cluster results.
