# SNIP-selected compact replacement with PEFT in one block

This experiment lives on branch `sft-snip-compact`. The original, compact and
subtraction branches are unchanged. Use `train_snip_compact.py` here, or the
repository-root `run.py snip-compact` route. The copied legacy trainer and
launchers retain their old behavior; they do not run this new experiment.

## What happens

1. Load the original pretrained ViT and preserve the existing dataset splits,
   augmentations, task classifier and sample budget.
2. Cache 64 training images for pseudo-inverse initialization. Build one
   substituted candidate for each of the 12 possible replacement positions.
   The default replacement is the existing compact symbolic filter; `lowrank`
   and `dense` are also available. It replaces the block's full output directly.
3. Train every surviving candidate for the same epoch budget on the same
   training order and augmentation seeds. Train its replacement, all remaining
   LayerNorms and the classifier; freeze the dense pretrained backbone.
4. On a fixed training calibration subset, compute per-parameter
   `abs(weight * gradient(mean cross-entropy))` and sum across the **whole
   substituted network** (paper equations 8 and 9). Temporarily differentiating
   frozen weights to measure sensitivity does not train them. Accumulate
   gradients across microbatches before taking the absolute product.
5. Update each candidate's score using equation 10,
   `q = momentum*q + (1-momentum)*network_snip`. Prune the lowest-scoring
   candidates each epoch. Default progression: **12 → 6 → 3 → 2 → 1**.
   Retain optimizer moments and reuse the winning candidate's trained state.
6. Score blocks in the winning substituted model, before adding adapters.
   Exclude the replaced position and select the **highest-SNIP surviving
   original block**. Ties use the lower original index.
7. Add PEFT to that one block's QKV, attention projection and two MLP linears.
   Train the compact filter, that block's adapters/selected columns, all
   LayerNorms and the classifier. All other dense pretrained weights stay frozen.
8. Select the final checkpoint by validation accuracy, restore it and evaluate
   test accuracy. Neither validation nor test examples enter SNIP selection.

```text
Training calibration → 12 compact-substituted models
                         ↓ equal training + whole-model SNIP + EMA pruning
                       Winning compact position
                         ↓ per-block SNIP in the winning model
                       Highest surviving block → PEFT only here

Final model: frozen dense blocks + compact filter + one PEFT block
             all LayerNorms and task classifier also trainable
Validation → best checkpoint → test evaluation
```

The compact block is not removed, and the highest-score block is not fully
fine-tuned. LoRA/DoRA learn changes to its effective weights; PaCA trains only
selected columns; fused PaCA learns LoRA/DoRA changes in those columns. DoRA's
magnitude parameters are trainable. RPaCA resamples only that block and clears
optimizer moments for the reset parameters. Uni-LoRA and Uni-DoRA are excluded.

## Run

From the repository root:

```bash
python3 run.py snip-compact --dataset pets --seed 18 --device cuda \
  --filter-type symbolic --filter-rank 16 --adapter-type lora --lora-rank 16

python3 run.py snip-compact --dataset cifar100 --seed 18 --device cuda \
  --adapter-type dora --lora-rank 16 --epochs 100

python3 run.py snip-compact --dataset pets --seed 18 --device cuda \
  --adapter-type paca --paca-rank 32 --paca-tuner lora --paca-adapter-rank 16

python3 run.py snip-compact --help
```

Supported families: `lora`, `dora`, `paca`, `rpaca`. PaCA/RPaCA tuners:
`direct`, `lora`, `dora`. Supported standalone LoRA/DoRA initialization:
`default`, `loftq`. Orthogonality penalties work for standalone and fused
LoRA/DoRA, with `--lora-ortho-lambda1` and `--lora-ortho-lambda2`.

The default classification budget is early stopping:
`--epochs -1 --max-epochs 200 --patience 10`. Use `--epochs 100` for fixed epochs.
Both use cosine scheduling and restore the best validation checkpoint.
SNIP search is an additional, separately reported budget. It trains 23
candidate epochs with default settings, then begins final PEFT training.

Search controls: `--snip-samples 64 --snip-batch-size 32
--snip-search-epochs 4 --snip-search-lr 0.001 --snip-momentum 0.9
--snip-prune-fraction 0.5`. A budget that cannot reduce all 12 positions to
one is rejected. At least 64 training images are required. All original 13
datasets are supported; the default selected training-pool budget is 1,000
images, not the full dataset. Use `--use-full-dataset` explicitly if intended.
All compared PEFT runs should use identical dataset, seed, compact and search
settings. Each invocation performs its own search; matching those settings
makes the search independent of the subsequently selected adapter family.

Direct invocation if all Python files are together in this folder:

```bash
python3 train_snip_compact.py --dataset pets --adapter-type lora \
  --lora-rank 16 --filter-type symbolic --device cuda
```

## Cluster and outputs

The integrated A100 launcher already accepts the new workflow:

```bash
mkdir -p /export/home/achyut/Sarvesh/sflogs
# Submit from the extracted repository directory containing run.py.
sbatch slurm/run_integrated_a100.sh snip-compact --dataset pets \
  --adapter-type lora --lora-rank 16 --filter-type symbolic --seed 18 --device cuda
```

For B200 use `slurm/run_integrated_b200.sh` with the same arguments. Set
`CODE_DIR` to the extracted repository root if submitting elsewhere. These
launchers retain your cluster resources and Conda activation.

The root CLI writes `outputs/snip-compact/<unique-run>/` containing:

- `metrics_summary.json`: validation/test accuracy, replacement and PEFT
  indices, configuration, parameter accounting and timing;
- `snip_search.json`: candidate scores, EMA updates, survivors, search resources,
  final per-block scores and initialization report;
- `snip_block_scores.csv`: per-block scores and replacement/PEFT markers;
- `best_model.pt`, `history.csv`, training/learning-rate/parameter plots;
- `filter_equations.json`: trained compact filter equations/representation.

Direct invocation defaults to `outputs/` beside `train_snip_compact.py`.
Use `--data-dir` for an existing cache and `--output-dir` to change the result
root. Completed configurations are protected from accidental overwriting;
`--overwrite` is explicit. Concurrent writers to the same run are rejected.

```bash
python3 run.py collect-all
# compiled_results/all_results.csv includes the new workflow and PEFT index.

# Optional: train with --defer-test, then evaluate the saved checkpoint later.
python3 run.py evaluate-snip-compact --summary /absolute/path/metrics_summary.json --device cuda
```

## Paper alignment and limits

The candidate-network saliency, EMA and lowest-score pruning implement the
mechanism described on page 4 of the uploaded SFP paper, equations 8–10.
This is different from simply replacing the original block with the lowest
individual block score. LayerNorm adaptation follows the paper's stated
exception to freezing; the existing task classifier training is preserved.

The compact symbolic replacement and PEFT of the highest-score surviving block
are **your extensions**, not methods specified in that paper. The article does
not specify every search hyperparameter; momentum, prune fraction and search
budget above are explicit implementation defaults. Dataset and training choices
remain those of the existing reviewed code, so this is not a claim of complete
paper reproduction or guaranteed accuracy. The symbolic filter remains the
existing bounded operator dictionary with a learned low-rank correction.
Trainable parameter percentage and total retained model size are different;
adding PEFT changes the former, so do not reuse the old 0.11% claim without
checking the new summary.

CPU tests exercise real gradients, candidate training/pruning, optimizer
continuity, PEFT scope, frozen weights, checkpoint restoration and test
evaluation on a tiny ViT. Full ViT training and A100/B200 accuracy still require
your cluster run.
