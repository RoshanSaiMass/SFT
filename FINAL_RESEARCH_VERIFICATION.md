# SFT_FINAL_RESEARCH verification

Base: `sft-loo-all-blocks-rank4`, commit
`35863364c19e79b94e9162891386d1bdc0abf5af`.

## Integrated workflows

The five existing backend folders are unchanged. The sequential SNIP backend is
copied from the tested one-candidate-per-epoch implementation without changes to
its mathematical source. A new scheduler/planner routes the seven primary
experiment types and the all-block LOO and SNIP compact PEFT extras to local
repository backends. IA3 is not implemented.

Ranks are persisted in campaign manifests and exact task commands: compact
filter, independent LoRA/DoRA, PaCA/RPaCA selected columns, fused adapter rank and
alpha. By default, alpha is twice the actual adapter rank. Fused rank limits,
finite settings and sequential SNIP's 11-round minimum are validated.

## Executed checks

- Root suite: **158 passed**, including 46 new scheduler checks covering default
  and modified backend CLI arguments, method/dataset filtering, rank validation,
  automatic chunking, and helper discovery from repository, scheduler and Slurm
  spool contexts with a stale legacy `SUITE_DIR`.
- Sequential SNIP: **40 passed**, exercising score gradients/restoration,
  one-candidate elimination, surviving optimizer state, dense/compact variants,
  adapter scopes and checkpoint restoration.
- Additional compiler fixture: passed; included the new all-block LOO and SNIP
  PEFT groups, selected the block with highest validation accuracy despite a
  different highest-test block, and wrote CSV/HTML plus PNG/SVG/PDF figures.
- Submission was rechecked after adding explicit scheduler environment exports
  and default evaluation of validation-selected removal checkpoints.
- Bash syntax checks, `git diff --check`, previous-backend byte checks and
  sequential-SNIP source byte checks passed.

## Operational details

Seven main grids contain 7140 training tasks per seed at default settings. Extra
all-block LOO and SNIP compact PEFT grids contain 12 and 240 tasks respectively.
Dataset/method filtering changes counts; the master submitter uses the actual
count and creates appropriately sized arrays/chunks. Direct headers retain the
default counts and need explicit overrides when changing grid dimensions.

Log directories are created before master submission. Independent submissions
require `sflogs` to exist first. The helper is sourced automatically and should
never be submitted as a job. Results, shared caches and campaign manifests are
created automatically. Compilation reports missing/failed tasks and evaluates
only validation-selected removal checkpoints from complete sweeps by default.

## Limits

GPU training, live Slurm scheduling, cluster dataset availability and improved
test accuracy are not established by CPU checks. Trainable percentages include
the classifier and vary with dataset class count. SNIP uses an explicit
one-removal schedule; exact paper reproduction is not established.
