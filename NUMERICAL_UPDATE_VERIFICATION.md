# SNIP numerical update and expanded targeted LOO

This update changes the combined `SFT_FINAL_RESEARCH` branch. Other published
branches and the original dense SFT/PEFT backend are not rewritten.

## Scientific changes

- Dense SNIP candidate initialization uses a float64 SVD with the singular-value
  cutoff derived from the precision of the captured features. Promoting float32
  observations no longer implicitly changes that cutoff to float64 epsilon.
- Candidate SNIP differentiates the same label-smoothed CE used in optimization.
  Original-model SNIP for optional targeted PEFT uses that loss convention too.
- Per-round gradient clipping respects `--grad-clip` when enabled.
- The ranking remains raw whole-network SNIP with one lowest-EMA candidate removed
  each round. No position is forced and no normalization reverses the ranking.
- Initialization reports every dense candidate's effective rank, weight norm,
  reconstruction MSE and CLS/patch errors. Scoring records CE, accuracy, saliency
  components, frozen/trainable contributions and post-training filter norms.
- Targeted LOO adds GELU, SiLU and Softsign after the existing six operator IDs,
  preserving previous checkpoint IDs. Maximum terms can be 1..9; nine allows 511
  subsets, not necessarily nine selected functions. Coefficients are allocated
  using the requested maximum and are included in parameter counts.
- The repository scheduler accepts these larger term limits for targeted LOO
  only. Single-group preparation/count/submission now honors `--group` as well
  as `--groups`, avoiding unwanted manifests for other backends.

The dense cutoff and frozen-weight scoring interpretation are documented choices;
the paper does not specify those implementation details. Compact SNIP filters
retain their separate initialization. No exact paper reproduction or improved
real-data accuracy is asserted. Use a fresh output/campaign for changed code.

## Validation

CPU environment: Python 3.12, torch 2.14.1, timm 1.0.30.

- SNIP search/integration and numerical regressions: **44 passed**, 20.33 seconds.
- Targeted LOO integration and symbolic expansion: **71 passed**, 9.06 seconds.
- Root integration/scheduler suite after the routing fix: **160 passed**, 78.67 seconds.
- Total across those isolated suites: **275 checks passed**.
- `git diff --check` passed. Nine-term LOO scheduler task preview and counts passed.

The numerical regression constructs ill-conditioned float32 observations and
confirms that the precision-aware fit stays near the float32 reference, while
the old promoted/default-float64 fit has a weight norm over 100 times larger.
Other tests cover exact smoothed-loss saliency, component-sum accounting, frozen
flags, per-candidate state, checkpoint round trips, and 129/381/501/511 searches.

No CUDA training, pretrained dataset accuracy reproduction, or actual cluster
queue execution was performed. The numerical test does not establish the cause
of the user's historical block-0 winner or guarantee that a new winner differs.
