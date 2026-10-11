# LOO implementation verification

Base: `sft-snip-compact` commit `214e43f6aa1af64dec97f7baf7a9baea660a1020`.
New branch: `sft-loo-compact`. No modifications to `SFT_ORIGINAL`,
`SFT_EXPERIEMTN`, `SFT_BLOCK_SUBTRACTION` or `SFT_SNIP_COMPACT`.
All supporting modules copied into `SFT_LOO_COMPACT` match source bytes.
The PEFT module SHA256 is
`7a0298914cfb5ce869a1803104942e87197e677c6732540e77a7bd0ac1903733`.

Environment: Python 3.12.14, PyTorch 2.14.1+cpu, timm 1.0.30, pytest 9.1.1.
`pip check` reports no broken requirements.

The isolated workflow verification completed with:

| Suite | Passed |
| --- | ---: |
| Root CLI/integration/launcher | 60 |
| Original | 121 |
| Existing compact experiment | 222 |
| Block subtraction | 167 |
| SNIP compact | 258 |
| New LOO compact | 126 |

One additional test then passed for a 12-block model, verifying all original
indices 0..11, 13 screening evaluations, and DoRA placement at block 11.
Thus **955 tests passed** across these checks. No failing/skipped tests remain.

New tests check:

- Each removal loss matches a separately copied one-block-deleted model.
- Identical images and example-weighted loss aggregation across batch sizes.
- Architecture, parameter flags, gradients and mixed module modes survive
  screening; failure paths also restore architecture and weights.
- Signed-loss ranking, negative deltas, tie handling and collision fallback.
- Head warm-up changes classifier weights only; zero warm-up is explicit.
- Deterministic class balancing, finite losses, and preserved calibration RNG.
- Identical rankings and warm-up histories across different PEFT families.
- Full training and prediction-preserving checkpoint reload for 11 adapter
  configurations (including no-PEFT and standalone LoftQ), with symbolic,
  lowrank and dense filters. Only the selected original block has adapters.
- Exact trainable scope, frozen dense weights, RPaCA checkpoint buffers,
  validation checkpoint restoration and deferred test evaluation.
- Completed-run protection, per-block CSV/selection JSON, full result CSV,
  real CLI subprocess help, and all eight A100 array task commands.

The Slurm script also passed `bash -n`. The launcher tests execute its shell
dispatch with environment activation and Python training stubbed; they verify
arguments and dataset/budget overrides, not actual cluster submission.

No CUDA device, downloaded target dataset, full pretrained training run or
Slurm scheduler was available in this verification environment. These checks
establish implementation behavior on synthetic CPU models, not accuracy,
paper reproduction, or guaranteed GPU/cluster execution.
