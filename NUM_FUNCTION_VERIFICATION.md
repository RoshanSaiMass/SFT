# Exact function-count experiment verification

Branch: `num_function`. Base: `SFT_FINAL_RESEARCH` commit
`28e30a39993d6026d507a9f016659ae486090e5d`.

The new standalone backend is `SFT_NUM_FUNCTION/train_num_function.py`.
Its `--num-functions N` enumerates distinct additive combinations of exactly N
functions. Counts 3,5,7,9 enumerate 84,126,36,1 candidates. Each candidate is
coefficient-fitted, evaluated on held-out training images, and the winning
combination is refitted. Requested and selected term counts are explicitly saved.

LOO screening, targeted PEFT, remaining LayerNorm/head training, and dataset splits
are retained. All six existing scientific backend folders match the base commit.
The maximum-term LOO method and its published branch were not rewritten.

CPU validation (Python 3.12, torch 2.14.1, timm 1.0.30):

- New exact-count backend: **24 passed**, 5.25 seconds.
- Root integration/scheduler/route suite: **165 passed**, 84.48 seconds.
- Total across those suites: **189 tests passed**.
- Actual trainer parser/validation accepted all 32 rank/count/LoRA-or-DoRA settings.
- Standalone scheduler shell syntax passed. All 192 LoRA mappings and all 192
  DoRA mappings were unique and correctly named; invalid task IDs were rejected.
- `git diff --check` passed.

Backend checks cover exact candidate enumeration, best held-out combination,
all-nine case, coefficient counts, gradients, checkpoint operator round trips,
full short training/evaluation, and targeted LoRA/DoRA/PaCA/RPaCA/fused/no-PEFT scope.
The standalone scheduler is supplied separately in chat and not installed in the
repository's scheduler suite.

These are compatibility and implementation checks, not GPU dataset accuracy
reproduction. More selected functions can have small coefficients and need not
improve accuracy. With nine functions selected there is one structure and only
coefficient fitting/projection training remain. No exhaustive permutation search
is needed because the additive expression is invariant to term order.
