# Run every block position with the uploaded experiment

`submit_block_sweep.sh` works with the original `SFT-sft-experiment.zip`.
Download this shell file into its `SFT_EXPERIEMTN` folder; the existing Python files
and `experiment.sbatch` do not need updating. Invoke it using **bash** to create
the result/log folders before it submits the array.

```bash
cd /export/home/achyut/Sarvesh/SFT_FINAL/SFT_EXPERIEMTN
bash submit_block_sweep.sh --training-budget fixed
```

This submits **60 Pets jobs**, seed 18:

* Each individual transformer block position 0 through 11.
* Dense filter control.
* Low-rank filter, ranks 8 and 16.
* Symbolic filter, ranks 8 and 16.

Every task replaces **one block**, explicitly using `--pruned-block`; SNIP block
search is skipped. This produces a block-versus-accuracy comparison. It does not
replace all 12 blocks simultaneously. The launcher's array is `0-59%4`, allowing
at most four simultaneous jobs. All results use plain SFT with no PEFT adapters.

The existing batch file supplies `gpu-a100`, `node1`, one GPU/task, 220G,
11:59:59, one task and eight CPUs. Conda defaults to `sft_env`; override using
`CONDA_ENV`, or point `CONDA_SH` to your Conda initialization script.

Budget choices:

```bash
# Fixed 100 epochs: 60 jobs. Best-validation checkpoint restored before test.
bash submit_block_sweep.sh --training-budget fixed

# Early stopping, patience 10, maximum 200 epochs: 60 jobs.
bash submit_block_sweep.sh --training-budget early

# Both budgets: 120 jobs in one array.
bash submit_block_sweep.sh --training-budget both
```

Both budgets retain the existing cosine learning-rate schedule. Change the limits
with `--epochs`, `--max-epochs` and `--patience`. Limit concurrency if needed:

```bash
MAX_CONCURRENT=1 bash submit_block_sweep.sh --training-budget fixed
```

The default dataset cache is the `data` folder **beside** `SFT_EXPERIEMTN`, i.e.
`/export/home/achyut/Sarvesh/SFT_FINAL/data`. This reuses the Pets cache from your
completed runs. Override using `--data-dir`. Four concurrent first-time downloads
should be avoided; pre-populate caches or use concurrency one for new datasets.

If your extracted folder is elsewhere, set `CODE_DIR`:

```bash
CODE_DIR=/absolute/path/to/SFT_EXPERIEMTN bash submit_block_sweep.sh --training-budget fixed
```

Optional multi-block compression sweep:

```bash
# 55 jobs: replace 1..11 blocks, five configurations per count.
bash submit_block_sweep.sh --training-budget fixed \
  --block-counts 1 2 3 4 5 6 7 8 9 10 11
```

This tests the **number** of replacements using existing SNIP selection. It does
not enumerate every subset of blocks. Eleven is the maximum because one attention
block must remain. Choose either `--block-counts` or `--block-indices`.

The launcher accepts `--datasets`, `--seeds`, `--ranks` and `--block-indices`:

```bash
# All 13 datasets, all positions, default five configurations: 780 jobs per seed/budget.
bash submit_block_sweep.sh --training-budget fixed --datasets \
  pets svhn flowers102 dtd caltech101 cifar100 fgvc_aircraft eurosat sun397 \
  pcam clevr dsprites-loc dsprites-ori
```

The launcher prints an absolute sweep folder under
`SFT_EXPERIEMTN/outputs/block_sweep_<unique ID>/`. It contains the manifest, logs
and one task directory per run. Existing training result folders are preserved.
Each task uses the original experiment worker, which checks CUDA, propagates
training failures and records `task_status.json`.

Once completed, replace `YOUR_SWEEP_FOLDER` with that printed directory:

```bash
python3 collect_experiments.py --root YOUR_SWEEP_FOLDER \
  --out compiled_results/block_sweep_all.csv
```

The resulting CSV has one row per completed run, preserving the block position,
filter type/rank, accuracy, parameter counts and efficiency metrics. Select the
block using validation accuracy; report its test accuracy. Check task statuses
and SLURM logs for failed/missing jobs.
