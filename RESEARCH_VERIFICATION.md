# Verification for sft_final_res

The final branch consolidates original SFT/PEFT, compact filters, physical block
subtraction and LOO compact replacement. The dedicated `SFT_SNIP_COMPACT`
experiment, evaluator, launcher and previous source bundles are excluded.
Original selection helpers inside retained backends remain available.

## Source preservation

All four backend trees are byte-identical to the verified source snapshot
`a915b0215c04758e3c42f97ec3e59289f2145b56`:

- `SFT_ORIGINAL`
- `SFT_EXPERIEMTN`
- `SFT_BLOCK_SUBTRACTION`
- `SFT_LOO_COMPACT`

Their source tree identifiers and PEFT checksums are in `SOURCE_PROVENANCE.json`.
No split, data augmentation, filter math, optimizer/training implementation,
PEFT projection or checkpoint format was changed in these backends. Original
source branches remain intact. Historical source archives remain available on
their original branches; the final package has only its current download.

Changes are confined to packaging, root dispatch/collection, documentation,
tests and launch orchestration. The original-grid manifest now records
`runner_path` separately from `code_dir`; this fixes locating the root helper
when the trainer resides in `SFT_ORIGINAL`. Older standalone manifests retain
the previous runner-location fallback.

## Executed tests

Environment: Python 3.12.14, PyTorch 2.14.1+cpu, timm 1.0.30, pytest 9.1.1.
`pip check` reports no broken requirements.

| Suite | Passed |
| --- | ---: |
| Root integration, packaging and Slurm dispatch | 62 |
| Original SFT/PEFT | 121 |
| Compact filters | 222 |
| Block subtraction | 167 |
| LOO compact | 127 |
| **Total** | **699** |

Each backend suite runs in a separate process to isolate identical module names.
All tests completed successfully, with no failed or skipped tests. Bash syntax
checks passed for all 15 included shell/Slurm scripts.

Root checks verify:

- The dedicated SNIP experiment is absent, and its training/evaluation routes
  are rejected. All retained routes resolve to real scripts.
- Real training/evaluation CLI help, caller-relative paths, subprocess isolation,
  argument preservation and exit-status propagation.
- Saved summaries compile into CSV with workflow, replacement/removal/PEFT
  indices, best validation accuracy, test accuracy and deferred null results.
- The actual all-dataset LOO launcher produces all 104 unique combinations and
  104 separate output locations, uses the submission directory, and supports
  fixed-epoch overrides. Invalid array indices fail.
- The correlation launcher uses the highest valid adjacent-pair correlation,
  retained-block PEFT and explicit `--min-correlation -1`.
- The original submitter prepares 520 tasks for two budgets with the correct
  `SFT_ORIGINAL` trainer and root helper paths, including paths containing spaces.
- A simulated Slurm-spooled batch script uses the recorded runner path; an
  older manifest without the new field still dispatches correctly.

Backend tests retain their existing coverage of adapter placement, frozen dense
weights, actual optimization, symbolic/pseudoinverse fitting, ranking and removal,
best-checkpoint restoration, test evaluation, parameter counting and checkpoint
round trips. LOO tests cover original indices 0..11, 13 independent screening
evaluations and targeted DoRA placement at the last block.

## Scope of verification

Launcher tests stub environment activation and training calls; they do not submit
Slurm jobs. Backend tests use synthetic CPU data/small ViTs. No GPU training,
target-dataset accuracy, upstream dataset-download availability or cluster
scheduling was established. Existing per-backend reports are historical records
of earlier stages and do not add GPU validation to this final branch.

Research comparisons still require matched dataset splits, seeds, training
budgets and validation-selection rules. LOO head warm-up and correlation feature
distillation are additional measured costs. Trainable fraction, retained model
parameters and checkpoint storage remain distinct quantities.
