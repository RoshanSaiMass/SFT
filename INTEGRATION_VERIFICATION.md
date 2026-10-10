# Integration verification

The integration branch merges `sft-experiment` at `9cc4fc0` and
`sft-block-subtraction` at `49ecab8` onto original `main` at `0245af6`.
The original reviewed ZIP is materialized into `SFT_ORIGINAL` so extraction is
not required before running the original workflow.

## Executed checks

CPU environment: Python 3.12.14, PyTorch 2.14.1+cpu, timm 1.0.30.

| Suite | Passed |
| --- | ---: |
| Root common CLI / integration tests | 42 |
| Original SFT/PEFT and original-copy data regressions | 121 |
| Compact filter workflow | 222 |
| Physical block-subtraction workflow | 167 |
| Total | 552 |

Backend suites run in separate processes to avoid importing a similarly named
module from another workflow. The verifier passed the original, compact and
subtraction suites; the root suite was rerun after final common-CLI changes.
No tests were skipped or disabled to obtain these results.

Integration checks exercise:

- Every registered backend script exists and receives the requested help/flags.
- Paths supplied by the user resolve from the caller's directory, including
  paths with spaces and `--flag=value` syntax.
- PEFT options pass through without removing ranks, fused tuners, regularizers
  or epoch budgets. Compact defaults do not force adapters off when requested.
- Correlation routing selects the highest valid pair by default; an explicit
  minimum correlation overrides it. Conflicting workflow/method flags fail.
- Actual child process working-directory isolation and nonzero exit propagation.
- Real compact grid preparation with dense/low-rank/symbolic configurations and
  both fixed and early-stopping budgets.
- One CSV preserving original, symbolic and deletion results, including null
  deferred test values; no averaging or reranking is performed by this collector.
- Parsing the user's exact log format: task 0 can refer to block 9, validation
  90.50%, test 84.49%, epoch 19. Missing logs remain marked rather than invented.
- Long original CLI run names fit actual filesystem limits and remain distinct
  when late options differ, using the existing compact branch's hashing helper.

An original grid was also prepared through the unified CLI: 20 configurations x
13 datasets x both budgets = 520 tasks for seed 18. This was manifest generation,
not training or cluster submission. Its code directory is the integrated
`SFT_ORIGINAL` and cache/output paths resolve correctly.

`bash -n` passed for the new integrated A100, B200 and removal-array launchers.
The A100 scripts include a GPU request to meet the user's observed QoS minimum.
Cluster scheduling and GPU training were not exercised in this CPU environment.

## Source preservation

Git comparisons confirm `SFT_EXPERIEMTN` is identical to source commit `9cc4fc0`
and `SFT_BLOCK_SUBTRACTION` is identical to `49ecab8`. The PEFT module
`single_filter_lora.py` is byte-identical in all three folders. Original trainer
logic remains unchanged except for adding the dataset-cache CLI argument.
The integrated original loader has the already-tested RGB/Caltech metadata/cache
fixes, preserving original split indices. Original source branch refs are not
updated by publishing this new branch. The original copy also uses the compact
branch's existing long-run-name helper, without altering its training logic.

## Limits

These checks establish CPU development readiness, dispatcher behavior and tested
backend functionality. They do not establish CUDA compatibility on the user's
cluster, full-dataset accuracy, paper parity, novelty, or successful compression
at a given accuracy threshold. Existing source-specific verification reports
remain included for detail. Parameter counts, trainable fractions and retained
model sizes must be compared separately under matched configurations.

The committed download archive contains tracked source files and the documented
CLI; it excludes datasets, cached weights, checkpoints and outputs. It is created
from the source commit preceding the archive publication commit, avoiding a ZIP
containing itself.
