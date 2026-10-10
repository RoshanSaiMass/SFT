# New subtraction workflow verification

The new folder uses the reviewed original PEFT archive, not the compact symbolic
implementation. `single_filter_lora.py`, `train_sfp_lora.py` and all three original
test files were compared byte-for-byte against `SFT_LoRA_2-reviewed.zip` and match.
Archive SHA256:
`c82fdc921eb31f03e1e79c72308c29efe6be85fa92249fea0294653662b56901`.

## Executed checks

`python -m pytest -q tests`: **167 passed** (118 copied original tests, 49 new
subtraction/data tests). Python 3.12, CPU PyTorch 2.14.1 and timm 1.0.30.

The new checks cover:

- Physical deletion, preserved surviving modules and original-index mappings.
- Pearson calculation, scale/offset mismatch, degenerate outputs and threshold
  rejection; hook cleanup and preservation of calibration sampling RNG state.
- Reduced feature matching error while LoRA base weights remain frozen.
- Classification training and full-checkpoint restoration for both workflows
  across 12 configurations: LoRA, DoRA, their LoftQ initializations, direct
  PaCA/RPaCA, their LoRA/DoRA fused forms and Uni-LoRA/Uni-DoRA.
- LayerNorm scope, trainable-parameter accounting and no trainable dense backbone.
- Validation-only ranking across seeds, missing candidates, deferred testing,
  selected-checkpoint testing and output CSVs.
- Complete manifest method/seed/block coverage, task resume, failure reporting,
  and detection of an entirely missing method before test selection.
- Caltech101 class metadata, grayscale conversion, RGB transform pickling and
  preservation of original train/validation/test indices in both dataset modes.

Separate architecture checks instantiated the actual
`vit_base_patch16_224.augreg_in21k` without downloading pretrained weights:
all six PEFT families produced finite `(1,37)` logits for `(1,3,224,224)` images
after physical deletion. Checks included original block 0, block 11 and an
interior block. Each student had 11 blocks and zero trainable dense backbone
parameters under the original PEFT freezing policy.

`bash -n` passed for all three SLURM scripts. CLI help and manifest construction
were exercised locally; no jobs were submitted to the user's cluster.

## Parameter counts for Pets / rank-16 LoRA

Counts include frozen weights as well as trainable weights; classification head
has 37 outputs. They are architecture counts, independent of training accuracy.

| Architecture | Total parameters | Trainable parameters | Reduction vs original 12-block model |
| --- | ---: | ---: | ---: |
| Original ViT-B, no adapters | 85,827,109 | Depends on fine-tuning policy | 0% |
| Delete later block, LoRA only in retained block | 78,935,845 | 260,389 | 8.0292% |
| Delete one block, LoRA in all 11 survivors | 80,901,925 | 2,226,469 | 5.7385% |

Both students retain the other pretrained transformer blocks. Adapter parameter
fraction is not the complete model size. Adapters on all surviving blocks can
make the model larger than an SFP student with a small replacement filter; compare
actual counts for each matched configuration rather than assuming deletion is
always more compact than every replacement method.

## Limits of this verification

No CUDA training, full-dataset run, pretrained-weight download, accuracy gain,
paper equivalence or cluster allocation has been established here. Correlation
is a heuristic; feature matching learns an approximation of two-block behavior.
Classification and validation determine whether that approximation is useful.
The feature matching stage has additional training cost, recorded separately.
Single-block sweep selection identifies the best candidate among the tested
configurations and seeds, not a globally optimal architecture.

The existing original audit documents remain copied unchanged and apply to the
source SFT/PEFT implementation. This report covers the newly added workflow.
