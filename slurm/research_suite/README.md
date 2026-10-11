# Separate research Slurm campaigns

These launchers use your **existing cluster code folders**. They do not change
any training algorithm or copy the excluded SNIP experiment into `sft_final_res`.
You can download the standalone `downloads/SFT-research-slurm-suite.zip`, extract
it, and run the submitter from its root. Its helper files must stay beside the
scripts as packaged; copying an individual `.sh` alone is insufficient.

## Paths and defaults

Edit `slurm/research_suite/config.sh` once if any path differs:

| Backend | Python folder |
| --- | --- |
| Dense candidate SNIP | `/export/home/achyut/Sarvesh/SFT_BEST/SFT_SNIP_COMPACT` |
| Dense replacement / PEFT | `/export/home/achyut/Sarvesh/SFT_FINAL` |
| Symbolic replacement | `/export/home/achyut/Sarvesh/SFT_EXP/SFT_EXPERIEMTN` |
| LOO / targeted PEFT | `/export/home/achyut/Sarvesh/SFT_LOO/SFT_LOO_COMPACT` |
| Correlation / removal | `/export/home/achyut/Sarvesh/SFT_COEFF_RED/SFT_BLOCK_SUBTRACTION` |

Defaults: A100, node1, one GPU per array task, 220G RAM, 11:59:59, four CPUs,
`sft_env`, seed 18, random 800/200 training/validation split from 1,000 images,
batch size 32, **early stopping**, max 200 recovery epochs, patience 10. The data
directory is shared at `SFT_BEST/data`; locked preparation prevents concurrent
first-download races. Dataset acquisition can still fail if an upstream source
is unavailable. Use current versions of the existing trainers with these CLI
flags; the submitter checks their files, not the software installed on your cluster.

All groups use the same 12 datasets, preserving the previous exclusion of
Caltech101: Pets, CIFAR-100, Flowers102, DTD, SVHN, FGVC Aircraft, EuroSAT,
SUN397, PCam, CLEVR Count, dSprites location and dSprites orientation. These are
the existing loader tasks, not a complete official 19-task VTAB reproduction.

## Separate scripts and output folders

Each sweep trial starts independently from the pretrained model. A block sweep
replaces/removes **one** block per model, rather than modifying all twelve at once.

| Script | Submit group | Result folder | Training runs per seed |
| --- | --- | --- | ---: |
| `01_snip_only.sh` | `snip_only` | `snip_only` | 12 |
| `02_all_blocks_dense.sh` | `all_block_snip` | `all_block_snip` | 144 |
| `03_all_blocks_peft.sh` | `all_block_snip+peft` | `all_block_snip+peft` | 3,168 |
| `04_loo_peft.sh` | `loo` | `loo` | 240 |
| `05_all_blocks_symbolic.sh` | `looall` | `looall` | 144 |
| `06_correlation.sh` | `sub_correlation` | `sub_sft` | 264 |
| `07_removal_sweep.sh` | `sub_removal` | `sub_sft` | 3,168 |
| `compile_results.sh` | Scheduled automatically | `compiled_results` | No training |

Total: **7,140 training runs per seed**. This count excludes candidate-training
rounds within SNIP, classifier warm-up within LOO, correlation distillation, and
test evaluation of removal winners. One array task handles one final training
run; its SNIP/LOO scoring work remains inside that task.

Result root defaults to `/export/home/achyut/Sarvesh/research_results`.
The submitter creates all group directories, campaign manifests and log folders
**before** submitting jobs. Each task keeps a separate directory with its
command, status, checkpoint, metrics JSON, history CSV and original plots.
Shared campaign labels let the compiler compare the experiment groups.

## Launch each experiment

Run from the extracted suite root, or the final repo root:

```bash
# Preview the jobs and write reviewable manifests; does not submit jobs.
DRY_RUN=1 bash slurm/research_suite/submit_suite.sh --dry-run --campaign preview

# Independent launches, using one shared label for one set of settings.
bash slurm/research_suite/submit_suite.sh --campaign research18 --groups snip_only
bash slurm/research_suite/submit_suite.sh --campaign research18 --groups all_block_snip
bash slurm/research_suite/submit_suite.sh --campaign research18 --groups 'all_block_snip+peft'
bash slurm/research_suite/submit_suite.sh --campaign research18 --groups loo
bash slurm/research_suite/submit_suite.sh --campaign research18 --groups looall
bash slurm/research_suite/submit_suite.sh --campaign research18 --groups sub_correlation sub_removal

# Schedule every group and one compiler job after all arrays finish.
bash slurm/research_suite/submit_suite.sh --campaign all18

# Fixed recovery budget, with a different label so budgets stay separate.
bash slurm/research_suite/submit_suite.sh --campaign fixed18 --epochs 100

# Smaller arrays if the cluster has a lower MaxArraySize.
bash slurm/research_suite/submit_suite.sh --campaign small18 --chunk-size 256 --concurrency 2
```

The submitter divides arrays into at most 1,000 tasks by default; for example,
3,168 runs become chunks of 1,000 + 1,000 + 1,000 + 168, with correct global
task offsets. `--concurrency 4` is **per chunk**, not a global GPU limit across
all submitted arrays. Slurm allocates only available resources; the total
campaign queue is large. The cluster must permit at least the chosen chunk size
as an array length. A task can still hit its 12-hour wall-time limit.

Use `--no-compile` when submitting individual groups if you want just one final
compiler job later. Otherwise each invocation schedules its own compiler after
its submitted arrays finish; incomplete other groups are reported. To retry
specific missing tasks, use their manifest, group launcher, offset and task ID
from `_campaigns/<label>_jobs.json`. Matching completed tasks are kept.

Direct `sbatch` of a launcher uses its printed default range. **For the two
3,168-run groups, that range covers only the first 1,000 runs. Use the submitter
to schedule all runs.** For direct submission, first create
`/export/home/achyut/Sarvesh/sflogs`, then submit from the suite/repo root. Its
`SLURM_SUBMIT_DIR` locates the helper even when Slurm spools the shell script.

To preview one exact Python training command:

```bash
python3 tools/research_suite.py task --group 'all_block_snip+peft' --task-id 0 --dry-run
python3 tools/research_suite.py task --group loo --task-id 0 --dry-run --epochs 100
```

## What each method does

1. **`snip_only`:** dense filter, no PEFT, SNIP momentum **0.3**, 64 initialization
   images, 64 scoring images, four search rounds, 50% candidate elimination.
   The last three search settings remain implementation choices; the paper only
   explicitly specifies 64 images for pseudo-inverse initialization. This is
   a paper-inspired dense baseline, not verified exact paper reproduction.
2. **`all_block_snip`:** manually choose block 0..11, replace it with a single
   dense affine filter initialized by the existing 64-image Moore–Penrose
   pseudo-inverse fit, then train filter + LayerNorms + head. **No SNIP search.**
3. **`all_block_snip+peft`:** the same dense replacement, plus the existing
   adapters on the eleven surviving transformer blocks. Dense pretrained
   backbone weights stay frozen; filter, adapters, LayerNorms and head train.
   This grid excludes SFT-only and full fine-tuning runs.
4. **`loo`:** warm up the classifier for five epochs, screen original-block
   deletions on up to 512 cached training images, replace the minimum signed
   CE-loss-increase block with a symbolic filter, and apply PEFT only to the
   highest-loss-increase surviving original block. Filter, targeted adapters,
   LayerNorms and head train; other dense backbone weights stay frozen.
   Screening does not perform twelve separate recovery trainings.
5. **`looall`:** independently replace each position with the symbolic filter,
   rank 16, max two terms, penalty 0.001 and 2,048 search tokens. No PEFT and
   **no LOO scoring**; the folder name is the requested output label.
6. **Subtraction:** correlation selects the highest valid Pearson adjacent pair
   with threshold **−1**, deletes the later block, and adapts the earlier block.
   Five feature-distillation epochs precede classification. The removal sweep
   independently deletes each original block and applies PEFT to all eleven
   survivors. Removal candidates defer test evaluation; the compiler selects
   using validation and tests only the winners by default.

LOO classifier warm-up and correlation distillation are extra costs outside the
recovery epoch budget. Set `--head-warmup-epochs 0` or `--distill-epochs 0` through
the submitter if explicitly disabling those experimental stages.

## PEFT grid

The 22 configurations reproduce the requested old block-sweep families and
orthogonality coefficients, **not** the different small-lambda original-grid
defaults:

- LoRA / DoRA: rank 16, alpha 32; no orthogonality, or λA = λB = 0.9, or 1.5.
- PaCA / RPaCA direct: 32 columns, no orthogonality.
- PaCA+LoRA, PaCA+DoRA, RPaCA+LoRA, RPaCA+DoRA: 32 columns, adapter rank 16,
  alpha 32; no orthogonality, or λA = λB = 0.9, or 1.5.
- Uni-LoRA / Uni-DoRA: rank 16, shared subspace dimension 72,000, no orthogonality.

Targeted LOO uses the first **20**, excluding both Uni families as requested.
Standalone/fused A/B adapters alone receive orthogonality regularization;
direct-column and Uni configurations never receive incompatible λ flags.
LoftQ is not an additional dimension in this requested grid.

## Compile everything

The automatic compiler uses an A100 job because it can test validation-selected
removal checkpoints; CSV parsing and plots themselves use CPU. It runs after
`afterany` dependencies so failed tasks are reported rather than preventing
compilation of successful results. It will **not** choose a removal/block-sweep
winner when any of its twelve candidates is missing.

```bash
# Schedule a final compilation manually; create sflogs first for direct sbatch.
mkdir -p /export/home/achyut/Sarvesh/sflogs
sbatch slurm/research_suite/compile_results.sh --campaign research18

# CPU-only recompile saved metrics, without evaluating any checkpoint.
python3 tools/compile_research.py \
  --root /export/home/achyut/Sarvesh/research_results \
  --campaign research18

# Optional descriptive test results for EVERY removal candidate.
# Selection still uses validation; this performs many additional test passes.
sbatch slurm/research_suite/compile_results.sh \
  --campaign research18 --evaluate-all-removals
```

Outputs include:

- `all_results.csv`: every completed run, preserving raw metadata and normalized
  dataset, method, λA/λB, seed, block, PEFT target, validation/test accuracy,
  parameter counts, measured time/memory, checkpoint and source-summary paths.
- `selected_by_validation.csv`: validation-selected winners of complete block
  sweeps and original SNIP/LOO/correlation-selected runs.
- `selected_summary.csv`: means, number of test seeds and sample standard
  deviation when at least two test seeds exist; one seed has no estimated SD.
- `by_group/*.csv`, `test_accuracy_wide_<campaign>.csv`.
- `training_history_long.csv`, `loo_scores_long.csv`, `snip_scores_long.csv`.
- `coverage.csv`, `incomplete_selections.csv`, `problems.json`.
- `figures/`: block-vs-validation/test curves, selected-test heatmap,
  accuracy-vs-trainable-parameter plots and selected-run learning curves,
  exported as **PNG, SVG and PDF**.
- `index.html`: local links to all tables and figures.

Missing test accuracy stays blank. The compiler does not invent per-epoch test
accuracy, FLOPs, latency, confidence intervals or missing original-model
parameter counts. The original dense trainer's compensation reference counts
only target linear weights; it is not an exact original whole-model count.
Comparisons stay separated by campaign and method; the collector never selects
the best test result. Old results outside these suite manifests are not mixed in.

Scripts were checked on CPU with synthetic saved results and actual CLI parsers.
No full dataset training or Slurm submission was executed in the cloud workspace.
