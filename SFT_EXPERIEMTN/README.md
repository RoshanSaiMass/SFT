# SFT_EXPERIEMTN: compressed filter research

This directory is self-contained. It starts from the reviewed stage-two SFT code
and adds filter compression experiments. The existing archives and root SLURM
files are unchanged. Use this folder's trainer, output folders and collector.

The goal is to test whether smaller replacements can preserve classification
accuracy. Parameter savings are counted; competitive accuracy is **not established**.
This is an extension of the existing SFP implementation, not a reproduction of a
symbolic-regression paper or a claim of matching the SFP paper's results.

## Implemented replacements

* `dense`: the existing affine filter, fitted by `X^+Y` on 64 training images.
* `lowrank`: fit the same matrix, take its truncated SVD and store two thin
  matrices. With embedding dimension `d` and rank `r`, parameters are `2*d*r+d`.
  This approximates the fitted matrix; exact inheritance is lost at reduced rank.
* `symbolic`: preserve the identity path and fit a low-rank residual. Search a
  bounded expression library in the projected feature space, then train its
  coefficients and projection matrices with the task loss. Parameters are
  `2*d*r+d+max_terms*r`. Projection matrices count toward storage and parameters.

Symbolic equations have the form

```
z = x @ P / scale
y = x + (scale * sum_t coefficients[t] * f_t(z)) @ Q + bias
```

The library contains linear, square, cube, tanh, sin and constant terms; square
and cube clamp their inputs to [-3, 3]. The search enumerates up to three distinct
terms with one shared expression structure and separate coefficients per latent
coordinate. It is a bounded symbolic-regression experiment, not unrestricted
TuringBot/genetic search. It cannot express arbitrary attention or arbitrary
multivariable equations. Dense projections remain necessary in this design.

The 64 calibration images come exclusively from the training loader. They are
split 75/25 **by image** for expression fitting/selection. Each side uses at most
`--symbolic-search-tokens` tokens (default 2048), with a local seeded sampler.
Candidate score is held-out residual MSE divided by identity residual MSE, plus
`--symbolic-penalty * number_of_terms`. After selection, coefficients are refitted
using tokens from all 64 training calibration images; the search basis stays fixed.
Dataset validation and test images are excluded from equation discovery.

The best validation checkpoint is restored before test evaluation, as in the
reviewed trainer. Equation structure remains fixed while projections/coefficients
fine-tune. `filter_equations.json` exports restored coefficients, scales, selected
operators and calibration diagnostics; the matrices are in the model checkpoint.

## Compression limit

Large **whole-model** savings require replacing several transformer blocks.
Shrinking just one existing filter saves under 1% relative to the single-filter
SFP model. The default grid tests 1, 2, 4, 6 and 8 replacements, including dense
controls for each count. Up to **11** of the 12 ViT-B blocks may be replaced.

Replacing all 12 is rejected: these filters operate on each token independently.
With no remaining attention, the CLS token cannot acquire image-patch information
and its predictions become image-independent. Keep at least one attention block.
Keeping one block still gives no accuracy guarantee.

`verification/parameter_report.csv` contains actual architecture parameter counts
for a 37-class ViT-B (pretrained weights and datasets were not downloaded):

| Replaced blocks | Dense model parameters | Symbolic rank-16 parameters | Reduction vs original ViT |
| --- | ---: | ---: | ---: |
| 1 | 79,329,829 | 78,764,613 | 8.23% |
| 4 | 59,837,989 | 57,577,125 | 32.91% |
| 8 | 33,848,869 | 29,327,141 | 65.83% |
| 10 | 20,854,309 | 15,202,149 | 82.29% |
| 11 | 14,357,029 | 8,139,653 | 90.52% |

These are parameter counts, not accuracy, throughput or GPU-memory measurements.
The CSV also reports savings relative to dense filters at the **same** replacement
count. For rank 16, one symbolic filter has 25,376 learned parameters, a 95.70%
reduction from the 590,592-parameter dense filter. Buffers and serialized checkpoint
bytes are reported separately in actual training summaries.

## Setup and individual runs

Use your working CUDA environment (for example `sft_env`), then install missing
dependencies using `pip install -r requirements.txt`. For local checks, install
`requirements-dev.txt`. This environment was checked on CPU only.

From this folder:

```bash
# Dense control, fixed 100 epochs, original SNIP selection.
python3 train_sfp_lora.py --dataset pets --mode sft --filter-type dense --epochs 100

# Symbolic replacement of four SNIP-selected blocks.
python3 train_sfp_lora.py --dataset pets --mode sft \
  --filter-type symbolic --filter-rank 16 --num-filter-blocks 4 --epochs 100

# Same exact blocks for controlled comparisons; accepts a single index too.
python3 train_sfp_lora.py --dataset pets --mode sft \
  --filter-type lowrank --filter-rank 16 --replace-blocks 1,3,5,7 --epochs 100

# Early stopping with cosine schedule and best-validation checkpoint restoration.
python3 train_sfp_lora.py --dataset pets --mode sft \
  --filter-type symbolic --filter-rank 16 --num-filter-blocks 4 \
  --epochs -1 --max-epochs 200 --patience 10
```

Default data/results paths are inside this directory even when launched elsewhere.
Use `--data-dir /existing/cache/path` to reuse your cluster dataset cache and
`--output-dir /new/results/path` to choose another experiment output location.
Existing dataset split algorithms and subset index selection remain unchanged.
Filter constructors preserve the CPU sampling RNG stream across architectures.
The copy fixes Caltech class metadata/RGB handling, adds missing data dependencies,
and bounds run-folder names to avoid long-filename failures. These fixes are local
to this experimental folder. PEFT implementation is copied unchanged.

Existing LoRA, DoRA, PaCA, RPaCA and Uni flags are available in this trainer. For
example, omit `--mode sft` to enable adapters:

```bash
python3 train_sfp_lora.py --dataset pets --adapter-type dora --lora-rank 16 \
  --filter-type symbolic --filter-rank 16 --num-filter-blocks 4 --epochs 100
```

Full fine-tuning and parameter compensation are incompatible with compressed
filters. Stacked dense filters/extra dense residual branches are also rejected
for compressed filters because they defeat this parameterization.

## A100 SLURM grid

Place this folder at `/export/home/achyut/Sarvesh/SFT_FINAL/SFT_EXPERIEMTN`.
The launcher uses `gpu-a100`, `node1`, one GPU, 220G and 11:59:59 per task.
`CONDA_ENV` defaults to `sft_env`; `CONDA_SH` may point to `conda.sh` explicitly.

```bash
# 25 tasks: Pets, seed 18, five replacement counts and five filter configurations.
bash submit.sh --training-budget fixed

# Same 25 configurations, early stopping.
bash submit.sh --training-budget early

# Both budgets: 50 tasks.
bash submit.sh --training-budget both

# Choose datasets/seeds/counts/ranks; each multiplier changes the task count.
MAX_CONCURRENT=2 bash submit.sh --training-budget fixed \
  --datasets pets cifar100 --seeds 18 42 --block-counts 1 2 4 --ranks 8 16 \
  --data-dir /export/home/achyut/Sarvesh/SFT_FINAL/data
```

The default grid is plain SFT, with dense controls, low-rank ranks 8/16 and symbolic
ranks 8/16. It does not multiply the grid across every PEFT configuration.
To relocate the folder, set `CODE_DIR` for `submit.sh`. Manifest/log folders are
created before submission. Each task has isolated results, a command manifest
and `task_status.json`; failed jobs propagate a nonzero exit code. Default
concurrency is one to avoid simultaneous first-time dataset/model downloads.
Higher concurrency should use a pre-populated shared dataset/model cache.

After completion, use the sweep root printed by the launcher:

```bash
python3 collect_experiments.py --root outputs/sweep_REPLACE_ME \
  --out compiled_results/experiments_all.csv
```

The CSV preserves each run's dataset, seed, filter choice, rank, block indices,
validation/test accuracy, parameter counts, compression ratios, checkpoint size
and efficiency metrics. Compare matched budgets/seeds and select block/rank/term
settings with validation data; reserve test scores for final reporting.
Failed/missing runs are absent from the CSV; inspect task statuses and SLURM logs.

## Verification

All 206 CPU tests passed. See `verification/REPORT.md` for results and limitations. No cluster job was
submitted and no real-data accuracy claim is made. For scientific comparison,
run the dense controls and compressed candidates on identical splits/budgets,
then repeat promising configurations across several seeds.
