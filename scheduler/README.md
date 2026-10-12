# Final research schedulers

These launchers use the backends in this repository. They do not depend on the
older scattered `SFT_BEST`, `SFT_EXP` or `SFT_FINAL` source folders. Their shared
dataset cache still defaults to `/export/home/achyut/Sarvesh/SFT_BEST/data` and
results to `/export/home/achyut/Sarvesh/research_results`.

Keep `scheduler`, `tools` and the backend folders together. `research_common.sh`
is sourced automatically, never submitted as a job. Helper discovery supports
submission from the repository root or scheduler folder, Slurm's spooled-script
layout and `SCHEDULER_DIR=/absolute/repo/scheduler`. An old `SUITE_DIR` is ignored.

## Seven main types

| Launcher | Group | Default tasks |
|---|---|---:|
| `01_snip_only.slurm` | `snip_only` | 12 |
| `02_all_blocks_dense.slurm` | `all_block_snip` | 144 |
| `03_all_blocks_peft.slurm` | `all_block_snip+peft` | 3168 |
| `04_loo_peft.slurm` | `loo` | 240 |
| `05_all_blocks_symbolic.slurm` | `looall` | 144 |
| `06_correlation.slurm` | `sub_correlation` | 264 |
| `07_removal_sweep.slurm` | `sub_removal` | 3168 |

Extras: `08_loo_all_peft.slurm` (rank-4 LoRA on all surviving blocks, 12 tasks)
and `09_snip_compact_peft.slurm` (compact-filter candidate SNIP and original-block
SNIP for targeted PEFT, 240 tasks). SNIP eliminates exactly one candidate per
search epoch: 12 positions to one in 11 rounds. This does not establish exact
paper reproduction or identical replacement positions.

The default seven arrays cover 7140 tasks per seed. They use the same 12 datasets
as the existing suite, excluding Caltech101. `--datasets` may select any supported
dataset, including Caltech101. Main PEFT grids contain 22 configurations; LOO and
SNIP compact PEFT exclude Uni methods, leaving 20. All-block LOO defaults to LoRA
only; `--methods` can explicitly select additional inherited adapters.

## Rank flags

| Flag | Meaning |
|---|---|
| `--filter-rank` / `--compact-rank` | Low-rank/symbolic replacement rank; default 16 |
| `--lora-rank` | Standalone LoRA rank; default 16, or 4 for all-block LOO |
| `--dora-rank` | Standalone DoRA rank; defaults to the selected LoRA rank |
| `--paca-rank` | PaCA/RPaCA selected columns; default 32 |
| `--paca-adapter-rank` | Fused LoRA/DoRA rank; otherwise uses their respective rank |
| `--lora-alpha` | Override adapter scaling; otherwise alpha = twice each adapter rank |

Dense filters retain their fixed 768-wide affine map; `--filter-rank` affects
compressed filters only. Fused adapter rank must not exceed PaCA column count.
The original orthogonality alternatives 0.9/0.9 and 1.5/1.5 are retained.

Other flags include `--compact-type dense|lowrank|symbolic`, `--loo-samples`,
`--head-warmup-epochs`, `--symbolic-max-terms`, `--symbolic-penalty`,
`--symbolic-search-tokens`, `--snip-momentum`, `--snip-search-epochs`,
`--snip-samples`, `--snip-filter-type`, `--epochs`, `--max-epochs`, `--patience`,
`--seed`, `--methods`, `--datasets`, `--campaign`, `--results-root`,
`--chunk-size` and `--concurrency`.

## Submit

The existing `sft_env` CUDA environment and datasets must be available. On the
login node, enter this repository's root and create logs before direct sbatch:

```bash
mkdir -p /export/home/achyut/Sarvesh/sflogs

# Seven main types, early stopping, original rank defaults.
bash scheduler/submit_all.sh --campaign research18

# One-block LOO: independently chosen LoRA/DoRA ranks and fused ranks.
bash scheduler/submit_all.sh --groups loo --campaign loo_ranks18 \
  --filter-rank 64 --lora-rank 32 --dora-rank 48 \
  --paca-rank 128 --paca-adapter-rank 64

# A small comparison: 3 datasets x 2 methods = 6 tasks.
bash scheduler/submit_all.sh --groups loo --campaign loo_small18 \
  --datasets pets cifar100 flowers102 --methods lora dora \
  --filter-rank 96 --lora-rank 32 --dora-rank 32

# Rank-4 LoRA across all 11 surviving blocks.
bash scheduler/submit_all.sh --groups loo_all_peft --campaign all_r4_18 \
  --filter-rank 16 --lora-rank 4

# Symbolic SNIP without PEFT, or SNIP with compact replacement and PEFT.
bash scheduler/submit_all.sh --groups snip_only --snip-filter-type symbolic \
  --campaign symbolic_snip18 --filter-rank 64
bash scheduler/submit_all.sh --groups snip_compact_peft --methods lora dora \
  --campaign snip_peft18 --filter-rank 64 --lora-rank 32 --dora-rank 32

# Fixed epochs and a command preview.
bash scheduler/submit_all.sh --groups loo --campaign fixed18 --epochs 100
bash scheduler/submit_all.sh --groups loo --campaign preview18 --dry-run
```

The submitter creates folders, saves rank/method/dataset manifests, splits large
arrays into chunks and schedules compilation after training finishes. `%4`
limits concurrency per array/chunk, not globally across the entire submission.
Use a different campaign name when changing ranks or other experiment settings.

### Independent submission

From the repository root (or scheduler folder), ranks can be passed directly:

```bash
sbatch scheduler/04_loo_peft.slurm --campaign direct18 \
  --filter-rank 64 --lora-rank 32 --dora-rank 32
```

Rank changes preserve task counts. Dataset/method selections change counts, so
prefer `submit_all.sh`; otherwise override the array size yourself. For example:

```bash
sbatch --array=0-5%4 scheduler/04_loo_peft.slurm --campaign small_direct18 \
  --datasets pets cifar100 flowers102 --methods lora dora --filter-rank 64
```

The default direct headers for groups 3 and 7 cover only 0..999. Use the master
submitter for all 3168 tasks, or submit successive chunks with `--offset`.
Independent submissions do not automatically schedule compilation:

```bash
sbatch scheduler/compile_results.slurm --campaign direct18 --evaluate-selected
```

## Saved results

Results are under `research_results/<group-folder>/<campaign>/`. Correlation and
removal use the shared `sub_sft` folder with separate subgroups. Training stores
checkpoints, metrics, histories and plots; task manifests include exact ranks
and commands. The compiler writes combined/per-group CSVs, validation-selected
results, seed summaries, coverage and failure reports, plots and an HTML index.
Removal tests are deferred until complete sweeps are selected by validation.
The master compiler evaluates selected removal checkpoints by default; use
`--no-evaluate-selected` on the submitter to leave those test values blank.

More capacity does not guarantee higher accuracy. Trainable percentages include
the task head and differ across datasets; frozen weights still occupy memory.
