# LOO symbolic replacement with rank-4 LoRA on every surviving block

This independent backend is on branch `sft-loo-all-blocks-rank4`. The existing
`SFT_LOO_COMPACT` backend still applies PEFT to one LOO-selected block. Its code,
dataset splits, symbolic-filter math and PEFT math are unchanged.

1. Load the original ViT-B/16 and warm up only its task classifier on training
   data for five epochs.
2. Independently bypass each original block and measure the signed change in
   classification loss on the same class-balanced, training-only calibration
   images. Select the minimum signed loss change, with lower indices winning ties.
3. Replace that one block with the existing rank-16 symbolic filter, initialized
   from 64 training images using pseudoinverse/SVD and bounded expression search.
4. Apply rank-4 LoRA to `qkv`, attention `proj`, `fc1` and `fc2` in **every one of
   the 11 surviving original blocks**. No adapters are installed in the filter.
   Alpha is 8, preserving the previous alpha/rank ratio of 2.
5. Train the symbolic filter, all 44 LoRA wrappers, surviving LayerNorms and
   classifier. Original backbone weights stay frozen. Assertions and saved
   parameter-name manifests enforce the scope.
6. Restore the best validation checkpoint, then evaluate test. Early stopping
   remains the default; `--epochs 100` selects fixed recovery training.

LOO chooses the replacement position only. The highest-loss block is still
reported for comparison, but does not restrict adapter placement. Saved results
contain `peft_block_indices` and the corresponding flags in `loo_block_scores.csv`.

## Run

```bash
python3 run.py loo-all-peft --dataset pets --device cuda --seed 18 \
  --filter-type symbolic --filter-rank 16 --adapter-type lora \
  --lora-rank 4 --lora-alpha 8 --loo-samples 512

python3 run.py evaluate-loo-all-peft --summary /path/to/metrics_summary.json --device cuda

python3 run.py collect-all --root outputs/loo-all-peft --out compiled_results/loo_all_rank4.csv
```

Additional inherited filter/adapter choices remain available for comparisons;
the provided Slurm launcher fixes the requested symbolic rank 16 and LoRA rank 4.
No full-backbone fine-tuning is enabled.

## A100 Slurm

Copy this complete `SFT_LOO_ALL_PEFT` folder into:

`/export/home/achyut/Sarvesh/SFT_LOO/SFT_LOO_ALL_PEFT`

The launcher is standalone and does **not** source `research_common.sh`:

```bash
mkdir -p /export/home/achyut/Sarvesh/sflogs
sbatch slurm/run_loo_all_peft_a100.slurm --campaign loo_all_rank4_seed18
sbatch slurm/run_loo_all_peft_a100.slurm --campaign loo_all_rank4_fixed18 --epochs 100
```

Each array has 12 tasks, one for each dataset from the existing research suite:
Pets, CIFAR-100, Flowers102, DTD, SVHN, FGVC Aircraft, EuroSAT, SUN397, PCAM,
CLEVR, dSprites location and dSprites orientation. Caltech101 remains supported
by the Python CLI, but is excluded from this launcher, as in the previous suite.
One A100 is requested per task, with at most four simultaneous tasks.

Results go under:

`/export/home/achyut/Sarvesh/research_results/loo_all_rank4/<campaign>/<dataset>/seed_<seed>/`

`CODE_DIR`, `DATA_DIR`, `RESULTS_ROOT`, and `CONDA_ENV` may be overridden in the
submission environment. The existing Python/Conda environment and required
dataset files must be available. The data loader retains its normal downloads.
`--dry-run` prints the selected task's command without activating Conda or training.

## Parameter budget and limits

With a 37-class Pets head, the original ViT has 85,827,109 parameters and this
configuration trains 629,829 (0.734% of the original). The replacement itself
has 25,376 parameters. Classifier size changes this percentage: the 397-class
SUN397 head makes the trainable share about 1.053%, so a uniform below-0.8%
claim across all datasets would be incorrect. Frozen weights still contribute
to inference/storage. No test-accuracy improvement is guaranteed.

CPU checks exercise 12-block screening, all-block adapter placement, frozen
weights, fixed/early-stopped training, checkpoint round-trips and parameter
counts. They do not establish GPU training accuracy or cluster execution.
