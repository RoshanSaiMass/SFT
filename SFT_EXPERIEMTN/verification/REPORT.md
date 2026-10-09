# Experiment verification

Checked on 2026-10-09 with Python 3.12, PyTorch 2.14.1+cpu,
torchvision 0.29.1+cpu and timm 1.0.30. CUDA was unavailable.

## Results

```
MPLCONFIGDIR=/tmp/peft-mpl XDG_CACHE_HOME=/tmp/peft-cache \
python -m pytest -q tests
206 passed in 14.65s
```

This includes the 118 reviewed-baseline tests and 88 additional experiment tests:

* Dense pseudoinverse inheritance and rank-deficient input behavior remain covered.
* Full-rank factorization matches the dense fit; lower ranks match truncated SVD.
* A known nonlinear teacher mapping is recovered using linear + square terms.
* Coefficients/projections have finite gradients and update during training.
* Checkpoint round-trips restore projections, coefficients, scales and operator IDs.
* Cached expression structure refreshes on state loading, avoiding per-forward
  CUDA/host synchronization for operator selection.
* Synthetic end-to-end trainer runs cover compressed filters with the existing
  PEFT integration cases, both single and sequential multiple replacement.
* Manual multiple-block choices work with dense/lowrank/symbolic plain-SFT runs.
* Invalid ranks/terms/penalties/block lists fail before dataset acquisition.
* Replacing every block is rejected. A separate tiny-ViT test demonstrates that
  token-local filters alone give image-independent CLS predictions.
* New-filter construction preserves the CPU sampling RNG stream.
* Caltech class count, grayscale conversion and original split index membership
  are verified with a synthetic dataset; RGB conversion remains picklable.
* Grid manifests have 25/25/50 default tasks for fixed/early/both budgets,
  distinct outputs and correct positive/-1 epoch flags. Dataset/seed multipliers,
  invalid counts and duplicate configurations are checked.
* CSV collection preserves per-run configurations, block indices and metrics.

`bash -n submit.sh experiment.sbatch` passed, and the trainer's `--help` completed.
No SLURM job was submitted.

## Counted parameter savings

`python parameter_report.py --out verification/parameter_report.csv` instantiated
an untrained 37-class ViT-B architecture and counted its parameters; it did not
download pretrained weights or data. The original model has 85,827,109 parameters.
The existing one-dense-filter SFP configuration has 79,329,829.

| Symbolic rank-16 replacements | Model parameters | Fewer vs original ViT | Fewer vs one-filter SFP |
| --- | ---: | ---: | ---: |
| 1 | 78,764,613 | 8.23% | 0.71% |
| 4 | 57,577,125 | 32.91% | 27.42% |
| 8 | 29,327,141 | 65.83% | 63.03% |
| 10 | 15,202,149 | 82.29% | 80.84% |
| 11 | 8,139,653 | 90.52% | 89.74% |

The CSV also includes ranks 8 and 32 and dense/lowrank controls. These counts
exclude PEFT adapters because this report's reference is plain SFT. Actual
training summaries account for any enabled adapters, buffers and checkpoint bytes.

## Isolation and limits

Baseline source provenance: `SFT_LoRA_2-reviewed.zip`, SHA256
`c82fdc921eb31f03e1e79c72308c29efe6be85fa92249fea0294653662b56901`.
The PEFT implementation file `single_filter_lora.py` is byte-identical to that
archive. Dataset split algorithms/index selection and original SNIP scoring are
retained. Changes are confined to the new `SFT_EXPERIEMTN` folder on branch
`sft-experiment`; existing archives/root SLURM files are unchanged.

CPU synthetic checks establish implementation behavior, not real-data accuracy,
paper parity, A100 performance or compatibility with every installed CUDA build.
The symbolic search is explicitly bounded and still stores dense thin projection
matrices. It does not replace the entire network with a few-byte formula.
The baseline SNIP proxy is inherited; it does not guarantee the best block set.

The default 25-task Pets grid compares dense controls against lowrank and symbolic
rank-8/16 filters for 1/2/4/6/8 replacements. Start with this grid, inspect validation
and test metrics, then extend promising settings/seeds. The rank/block choices are
research settings, not recommendations justified by observed dataset accuracy.
