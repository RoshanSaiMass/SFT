# Verification of the SNIP compact extension

Executed on 2026-10-10 in the Codex CPU environment: Python 3.12.14,
PyTorch 2.14.1+cpu, timm 1.0.30 and pytest 9.1.1.

| Suite | Passed |
| --- | ---: |
| Root CLI / collectors | 47 |
| Original SFT / PEFT | 121 |
| Existing compact workflow | 222 |
| Block subtraction | 167 |
| New folder, including inherited compact tests | 257 |
| **Total** | **814** |

Each suite ran in its own process. The new folder includes 35 new tests; its
other 222 tests verify the copied supporting implementation. Root tests also
exercise the new trainer/evaluator help and CSV collection. All tests completed
with no failures or skips. Source suites executed through
`python tools/verify_workflows.py`; after adding final tests, root and new-folder
suites ran again individually.

The new tests verify the saliency against gradients of the full calibration
loss, including uneven microbatches; score measurement restores gradient flags,
module modes and preexisting `.grad` values without changing model weights.
They check training-only calibration, per-candidate equal sample/step budgets,
EMA pruning, continued AdamW moments across epochs, deterministic data ordering,
highest surviving block selection and rejection of incomplete search budgets.

Twenty complete tiny-ViT training runs cover both symbolic and low-rank
replacement with ten adapter configurations: LoRA and DoRA with default and
LoftQ initialization, direct PaCA/RPaCA, and each PaCA/RPaCA LoRA/DoRA tuner.
They check four adapter wrappers in exactly one selected block, the exact
trainable parameter set, unchanged untouched dense weights, validation/test
metrics, symbolic/PaCA checkpoint reconstruction and repeated test evaluation.
A separate run checks early stopping and deferred test evaluation.

The three existing folders have no diffs against `sft-integrated` at `206ad14`.
The copied PEFT implementation remains byte-identical to the original:
`single_filter_lora.py` SHA256
`7a0298914cfb5ce869a1803104942e87197e677c6732540e77a7bd0ac1903733`.

These are CPU functional checks on synthetic small models. No full ViT-B/16
training, dataset accuracy, GPU execution or Slurm submission was performed.
Search hyperparameters not specified in the paper remain explicit defaults;
compact replacement and highest-block PEFT are experimental extensions.

## Original-model PEFT scoring clarification

The updated new-folder suite passed **257 tests** after separating the two
rankings. PEFT scores are now measured on the original pretrained ViT with the
existing dataset classifier, before any replacement or training. Tests compare
these saved scores with direct original-model gradients and verify that only
candidate-network scores control compact placement. An adversarial collision
test makes the original highest block equal the compact position, and confirms
that the next-highest surviving original block receives PEFT, even when compact
model block scores would choose a different target. The unchanged other-suite
results above remain from their preceding runs; they were not rerun for this
localized update. GPU accuracy is still untested.
