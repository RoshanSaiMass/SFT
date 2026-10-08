# A100 array for all supported datasets and methods

Copy the supplied `slurm` folder beside your existing Python files in:

```text
/export/home/achyut/Sarvesh/SFT_FINAL/
    train_sfp_lora.py
    requirements.txt
    slurm/submit_all.sh
    slurm/all_methods.sbatch
    slurm/sweep.py
```

The launcher defaults to this code folder. It submits one array to `gpu-a100`, `node1`, with one GPU, eight CPUs, 220G RAM and 11:59:59 per task. It activates `paca_env` and checks CUDA before training. No cluster job has been submitted or GPU run verified from the cloud review environment.

## Install once, then submit

Use the cluster's existing CUDA-compatible torch/torchvision installation in `paca_env`; install the project's remaining dependencies there once. The array does not run pip in every job.

```bash
cd /export/home/achyut/Sarvesh/SFT_FINAL
conda activate paca_env
python -m pip install -r requirements.txt scipy h5py gdown

# Review the complete manifest without submitting anything.
bash slurm/submit_all.sh --training-budget both --dry-run

# Submit one array covering both protocols.
bash slurm/submit_all.sh --training-budget both
```

The dry run creates its own sweep folder. The real submission creates a separate one and prints its manifest path and sbatch command. Log directories exist before sbatch is called. Submit through `submit_all.sh`, since it computes the array size and supplies the manifest; the batch file alone has no fixed array range.

## Select the training protocol

| Flag | Per-task training | Tasks with seed 18 |
|---|---|---:|
| `--training-budget fixed` | 100 epochs; cosine decay; restore best validation checkpoint | 260 |
| `--training-budget early` | Early stopping: patience 10, maximum 200 epochs; cosine decay; restore best validation checkpoint | 260 |
| `--training-budget both` | Separate tasks for each of the above protocols | 520 |

`fixed` is the default. Each protocol is recorded separately in the manifest, task folders and aggregate statistics. Test evaluation happens after best-validation checkpoint restoration. Early stopping is an additional stopping rule, not a replacement for cosine scheduling.

Examples for either protocol alone:

```bash
bash slurm/submit_all.sh --training-budget fixed
bash slurm/submit_all.sh --training-budget early
```

## Grid and settings

All 13 supported datasets are included: pets, svhn, flowers102, dtd, caltech101, cifar100, fgvc_aircraft, eurosat, sun397, pcam, clevr, dsprites-loc and dsprites-ori. Existing dataset proxies and split rules are retained. Each run requests 1,000 samples, giving the existing 800/200 train/validation subset where enough samples exist; the existing test split is unchanged.

The 20 named configurations are:

| Configuration IDs | Meaning |
|---|---|
| `sft`, `full_finetune` | Plain SFP baseline; full fine-tuning baseline |
| `lora`, `lora_loftq`, `lora_ortho` | SFP + LoRA: standard, LoftQ-style initialization, orthogonal regularization |
| `dora`, `dora_loftq`, `dora_ortho` | Corresponding SFP + DoRA configurations |
| `paca_direct`, `rpaca_direct` | SFP + direct selected-column adaptation |
| `paca_lora`, `paca_dora`, `rpaca_lora`, `rpaca_dora` | SFP + fused column adaptation with the selected tuner |
| `paca_lora_ortho`, `paca_dora_ortho`, `rpaca_lora_ortho`, `rpaca_dora_ortho` | Corresponding fused methods with orthogonal regularization |
| `unilora`, `unidora` | SFP + shared-vector adaptation |

These cover method families and named variants, not every combination of ranks or other hyperparameters. Except for full fine-tuning, each run uses the existing SNIP proxy to replace one block. This array does not sweep all 12 replacement positions. Adapter projection placement, family definitions and research qualifications are described in `PEFT_AUDIT.md`.

Defaults: seed 18, batch size 32; LoRA/DoRA rank 32 with alpha 32; PaCA/RPaCA 32 columns with fused tuner rank 4; Uni methods rank 4 and shared dimension 72,000. Orthogonal variants use both penalties at 0.0001. Other optimizer/filter options retain trainer defaults, including full fine-tuning's automatic learning-rate adjustment. Parameter budgets differ by method; counts and efficiency measurements are saved for comparison.

## Folders, caching and results

The helper creates:

```text
SFT_FINAL/
    data/                              shared dataset cache
    outputs/
        cache/                         shared pretrained/cache files
        sweep_<UTC timestamp>_<id>/
            manifest.json
            manifest.csv
            logs/<job_id>_<task_id>.out
            logs/<job_id>_<task_id>.err
            tasks/<id>_<dataset>_<config>_<budget>_seed18/
                data -> shared dataset cache
                status.json
                results/<trainer run name>/
                    best_sfp_lora_<dataset>.pt
                    metrics_summary.json
                    history.csv
                    *.png
            aggregate_summary.json
            aggregate_summary.csv
            aggregate_stats.csv
            aggregate_summary.xlsx
```

Each task has a unique working directory and output tree. Shared download locks prevent simultaneous initial acquisition of the same dataset or backbone; dSprites tasks share their archive lock. Ready markers appear only after preparation succeeds. Uncached preparation needs network access on the compute node. The launcher uses your existing dataset code. If a large download was interrupted, repair its incomplete archive/extraction before retrying; a ready marker alone cannot repair a damaged cache.

Aggregation runs automatically after each task. The last finishing task updates the final table; failed and waiting tasks remain visible, and statistics only use complete runs within the same dataset/configuration/protocol. You can refresh the table at any time using the exact path printed at submission:

```bash
python slurm/sweep.py collect --manifest /absolute/path/to/sweep_folder/manifest.json
```

Jobs killed externally can leave a `running`/`preparing` state; use that task's SLURM status and logs to distinguish a live job from a timeout/cancellation. Failed jobs return a nonzero exit code. The full metrics remain in individual `metrics_summary.json` files.

Misclassified-image dumping defaults to off for this large grid; checkpoints, metrics, histories, plots and logs are saved. Set `SAVE_MISCLASSIFIED=true` to also save all wrong predictions. The full grid needs substantial disk space: hundreds of full-model checkpoints plus large datasets and their extraction archives. Use your cluster's scratch storage for caches/results if needed.

## Optional overrides

```bash
# Choose additional seeds and allow more tasks when cluster resources permit.
SEEDS="18 42 123" MAX_CONCURRENT=2 bash slurm/submit_all.sh --training-budget fixed

# Put large caches/results on scratch without changing Python split logic.
DATA_DIR=/scratch/your_account/sft_data \
CACHE_DIR=/scratch/your_account/sft_cache \
OUTPUT_ROOT=/scratch/your_account/sft_results \
bash slurm/submit_all.sh --training-budget both

# Use a nonstandard Conda installation.
CONDA_SH=/absolute/path/to/conda/etc/profile.d/conda.sh \
CONDA_ENV=paca_env bash slurm/submit_all.sh --training-budget fixed

# Change budgets/batch size explicitly.
EPOCHS=100 MAX_EPOCHS=200 PATIENCE=10 BATCH_SIZE=32 \
bash slurm/submit_all.sh --training-budget both
```

`MAX_CONCURRENT=1` is the default. Additional seeds multiply the task count; the cluster must permit the resulting array range. `CODE_DIR` can override the source path, and `PREPARE_PYTHON` can choose the login-node interpreter used to create the manifest. Resource directives can be edited in `all_methods.sbatch`.

## Verification

The launch helpers passed 28 local tests, including all 20 method configurations against the actual trainer's CLI validation, task counts for both protocols, distinct result/statistics groups, quoted paths, child failures and collection. Both shell files pass `bash -n`. These checks do not submit a cluster job or verify a physical A100 run. The separately published PEFT code was checked read-only; no PEFT source changes accompany this launcher bundle.
