# Launchers for sft_final_res

Run integrated launchers from the repository root containing `run.py`, or export
`CODE_DIR` to that root. A100 logs use `/export/home/achyut/Sarvesh/sflogs`;
create this directory before submission. Integrated Conda default: `sft_env`.

| Launcher | Purpose |
| --- | --- |
| `run_integrated_a100.sh WORKFLOW FLAGS` | Any single A100 workflow |
| `run_integrated_b200.sh WORKFLOW FLAGS` | Any single B200 workflow |
| `run_correlation_a100.sh FLAGS` | Highest-correlation adjacent pair; Pets/LoRA default; no 0.9 cutoff |
| `run_loo_compact_a100.sh FLAGS` | Eight LOO PEFT configurations for one dataset |
| `run_loo_all_datasets_a100.sh FLAGS` | 13 datasets x eight LOO PEFT configurations = 104 tasks |
| `run_integrated_removal_array_a100.sh MANIFEST` | One physical-removal candidate per task; default array 0..71 |
| `submit_all.sh --training-budget fixed\|early\|both` | Original 13-dataset/20-configuration grid; automatic preparation/submission |

```bash
mkdir -p /export/home/achyut/Sarvesh/sflogs
sbatch slurm/run_correlation_a100.sh --epochs 100
sbatch slurm/run_loo_all_datasets_a100.sh --epochs 100
sbatch --array=0 slurm/run_loo_compact_a100.sh --epochs -1
sbatch slurm/run_integrated_a100.sh original --dataset pets --mode sft --pruned-block 3 --epochs 100 --device cuda
```

LOO/correlation default to early stopping (maximum 200 classification/recovery
epochs, patience 10). `--epochs 100` uses fixed epochs. Both restore the best
validation checkpoint. LOO has five classifier-only warm-up epochs; correlation
has five feature-distillation epochs, additional to the main budget.

The LOO arrays include LoRA, DoRA, PaCA, RPaCA, PaCA+LoRA/DoRA and
RPaCA+LoRA/DoRA, without Uni methods. The original grid additionally retains
Uni methods, SFT, full fine-tuning, supported LoftQ and orthogonal variants.
Ranks/parameter budgets differ by configuration and are saved in each summary.

The 104-task array groups eight methods per dataset: tasks 0..7 are Pets,
8..15 SVHN, 16..23 Flowers102, and so on. Its output root is
`outputs/loo_all_ARRAY_JOB_ID/DATASET/METHOD/`. Prepare shared dataset/backbone
caches before concurrent runs. `%4` permits up to four concurrent jobs and
remains subject to the cluster's actual GPU/RAM availability.

After completion:

```bash
python3 run.py collect-all
# Or collect only one array:
python3 run.py collect-all --root outputs/loo_all_JOBID --out compiled_results/loo_all_JOBID.csv
```

## Original grid

The original submitter now locates this repository's helper separately from
`SFT_ORIGINAL`, where the trainer resides. It creates logs before submission.
One seed gives 260 tasks for one budget or 520 for both.

```bash
bash slurm/submit_all.sh --training-budget fixed --dry-run
bash slurm/submit_all.sh --training-budget both
```

Overrides remain available: `SEEDS="18 42"`, `MAX_CONCURRENT=2`, `DATA_DIR`,
`CACHE_DIR`, `OUTPUT_ROOT`, `CONDA_ENV`, `CONDA_SH`, `PREPARE_PYTHON`, `EPOCHS`,
`MAX_EPOCHS`, `PATIENCE`, `BATCH_SIZE`, `NUM_SAMPLES` and `SAVE_MISCLASSIFIED`.
For this submitter only, `CODE_DIR` means the **original trainer folder**, not
the integrated root. It defaults to `REPO_ROOT/SFT_ORIGINAL`. Dataset/cache
roots default to shared `REPO_ROOT/data`, and results to
`REPO_ROOT/outputs/original_grid`. Manifests record `runner_path` separately.

Original-grid shared download locks and automatic result aggregation remain.
You can refresh its aggregate CSV with:

```bash
python3 run.py original-grid collect --manifest /absolute/path/manifest.json
```

## Removal arrays and compact sweeps

```bash
python3 run.py removal-sweep --prepare-only --dataset pets --seeds 18 --epochs 100 --device cuda
sbatch slurm/run_integrated_removal_array_a100.sh /absolute/path/manifest.json
# After completion, select with validation and evaluate winners:
python3 run.py collect-removals --root /absolute/path/removal_sweep_folder --seeds 18 --evaluate-selected --device cuda
```

Override the array range when seeds/methods/blocks change. Missing candidates
remain errors unless `--allow-partial` is explicitly chosen.

The compact all-position submitter remains in its backend:

```bash
CODE_DIR="$PWD/SFT_EXPERIEMTN" bash SFT_EXPERIEMTN/submit_block_sweep.sh --datasets pets cifar100 --ranks 16 --training-budget both
```

A100 launchers request a GPU, including for collection if run there, matching
the cluster's observed minimum-GRES requirement. Collection itself needs no GPU
unless it evaluates checkpoints. B200 retains `mig90g`, `gpu:3g.90gb:1` and 64G;
its log directory is `/home/achyutm01/SFT/sflogs`.

These launchers were checked locally; no cluster jobs are submitted by tests.
