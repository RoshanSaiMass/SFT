# Rank-4 all-block PEFT verification

Source base: `sft_final_res` commit `c24f48faa3aa9506c80310d80a8e43d4eec45d82`.
New branch: `sft-loo-all-blocks-rank4`.

## Preserved components

The four existing backend folders are unchanged. In the new `SFT_LOO_ALL_PEFT`
folder, `data.py`, `compressed_filters.py`, `single_filter_lora.py`,
`calibration.py` and `train_sfp_lora.py` are byte-identical to `SFT_LOO_COMPACT`.
Classifier warm-up, paired training-only LOO ranking, signed-loss tie rules,
symbolic fitting, data splits, PEFT math and recovery optimizer/scheduler are
preserved. The new workflow expands adapter placement from one original block
to every surviving original block.

## Executed checks

CPU runtime: PyTorch 2.14.1, timm 1.0.30, Python 3.12.

- New backend: **8 passed**, including 12-block training workflows, 44 LoRA/DoRA
  wrappers, PaCA/RPaCA/no-PEFT scope, frozen pretrained weights, replacement
  exclusions, missing-adapter rejection, checkpoint restoration, early stopping,
  deferred test evaluation and the real ViT-B/16 Pets parameter count.
- Root integration suite: **109 passed** initially; one new test had an incorrect
  command-list index. After correcting that test, its targeted rerun passed.
  All **110 root checks** have passed, including the two new CLI routes, CSV
  workflow identification, 12-dataset Slurm mapping and fixed/early budget cases.
- `git diff --check` and Bash syntax checking passed.
- Existing-backend and copied-math byte checks passed.

The parameter manifest and `peft_block_indices` record all surviving targets.
Checkpoint restoration checks that those indices match the injected adapters.
The single `peft_block_idx` field is null for this all-block workflow.

## Parameter budget

With Pets (37 classes), symbolic rank 16 and LoRA rank 4 across 11 original
blocks, trainable parameters total **629,829**, or **0.734%** of the original
85,827,109-parameter ViT. The task-head size varies by dataset; SUN397 is about
**1.053%**, so the below-0.8% result must not be generalized to every dataset.

## Limits

Tests use synthetic training data and CPU execution. GPU/Slurm execution,
dataset availability on the cluster and test-accuracy improvements are not
established by these checks. The provided array covers the 12 datasets from the
existing suite, excluding Caltech101; Caltech101 remains available through CLI.
