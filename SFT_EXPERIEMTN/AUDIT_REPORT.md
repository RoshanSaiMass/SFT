> Historical stage-one report. Adapter findings and CLI/diagnostic status are superseded by PEFT_AUDIT.md after the stage-two integration review.

# Focused SFP audit

Scope: use the attached checklist as a reference, subject to the user's newer instructions. Preserve the existing train/validation/test split algorithm, seeds, membership, sizes and dataset proxies. Leave LoRA, DoRA, PaCA, RPaCA, DoCA, UniLoRA and related adapter implementations for stage two. Do not build two new projects or impose every recommended default. The paper PDF itself was not supplied; paper claims below refer to the supplied checklist, not an independently verified PDF.

## Changes made

| Serious conflict or risk | Minimal correction | Reference |
|---|---|---|
| `blocks.1` substring also matched blocks 10 and 11, unintentionally training backbone weights and inflating filter parameter counts | Match exact `blocks.<index>.` prefix in freezing and reporting. Identify LayerNorm parameters by module type. | A49–56; freezing correctness |
| Validation shared the training dataset's random crop, flip, jitter and erasing | Recursively shallow-copy dataset/Subset views and attach the deterministic evaluation transform. All indices remain unchanged. | B2/B7; unstable checkpoint selection |
| dSprites horizontal reflection changes posX/orientation labels without relabeling | Remove horizontal flip only for dSprites pose tasks. CLEVR's count-safe flip is retained. | B2; label corruption |
| Backbone name did not explicitly select ImageNet-21K-only weights | Use `vit_base_patch16_224.augreg_in21k`, verified against installed timm metadata (21,843 pretrained classes). Set shared normalization to its published `(0.5, 0.5, 0.5)` mean/std. | A3; preprocessing parity |
| “pinv” initialization solved regularized normal equations in float32 | Use `torch.linalg.pinv(X.double()) @ Y.double()` in both filter initializers, retaining the correct transpose and existing filter architectures. | A32–35, A43; B5 |
| Initialization collected 11 batches, dependent on training batch size | Collect exactly 64 training images, slicing the final batch; reject insufficient data and always remove the hook. | A35 |
| Result plots chose replacement blocks by held-out test accuracy | Choose plots' blocks by mean validation accuracy. Retain diagnostic test maxima with explicit `exploratory_test_*` names. | Test-set selection leakage |
| Zero validation accuracy could leave no current checkpoint and reload a stale file from a previous run | Initialize best accuracy to negative infinity; only reload a checkpoint selected during the current run; use explicit device mapping and weights-only load. Stop on non-finite loss/validation metrics and reject nonpositive max epochs. | Checkpoint/run integrity |

Shared backbone, initialization, preprocessing and training guards affect runs using adapters as well. Adapter mathematics, initialization, injection, resampling, regularization, ranks and optimizer grouping have not been altered. Old checkpoints/results should not be mixed with new runs: backbone weights, normalization and SFP initialization deliberately changed. Analysis CSV diagnostic column names also changed.

## Important remaining differences: reported, not silently redesigned

- **Search is a proxy, not the checklist's SFP candidate-elimination search** (A76–89, B8). `snip_selection.py` scores original blocks once and picks low-scoring blocks. It does not train replacement candidates with equal resources, sum full-candidate saliency, maintain the specified moving average, or perform eleven eliminations. Implementing that is a substantial method change, not a surgical bug fix. Fixed paper-layer positions can be supplied via the existing zero-based `--pruned-block` option for single-block runs; this does not validate the search method.
- **Training is neither exact requested ES nor exact requested COS**: default early stopping plus cosine decay; positive epoch counts still reload a best-validation checkpoint. This is a defensible holdout-based experiment, not intrinsically leakage, because validation is not included in the training split. It does not reproduce the checklist's fixed 100-epoch final-model protocol (A58–60/B0/B7). Schedule and checkpoint-selection behavior are retained rather than changing the experimental design.
- **This is not a complete official 19-task VTAB reproduction**: dataset loaders use custom splits; dSprites is explicitly a proxy. The user's requested split preservation takes precedence. No benchmark-parity claim is made.
- Filters retain their bias and optional multilayer/nonlinear residual extensions. No original replaced transformer block remains as an active parallel path. These extensions were not redesigned.
- No separate warm-start head, exhaustive candidate search, full CKA/FLOPs/timing reproduction or three-seed benchmark was added.
- Low-impact CLI issue left unchanged: `train_sfp_lora.py --help` raises a formatting error from the literal `1%` in the minimum-LR help text. Ordinary argument parsing and the synthetic training run work.
- SNIP temporarily enables gradients without restoring prior flags; current training paths subsequently call freezing before optimizer construction. Standalone use needs care. Not modified in this focused pass.
- Exploratory test maxima remain diagnostics, not a permissible way to choose the reported configuration. Mean results across a block sweep are not equivalent to a benchmark result for a validation-selected configuration.

## Validation

- `python -m pytest -q tests/test_audit_fixes.py`: **8 passed** on CPU.
- Tests cover rank-deficient pseudoinverse/transpose, single/multilayer fitting, exact block-boundary freezing/counts, unchanged split membership in both subset/full modes, deterministic validation views, dSprites augmentation, exactly 64 init images and hook cleanup, validation-based reporting, and checkpoint normalization metadata.
- Ran the actual `train_sfp_lora.main()` with only dataset acquisition and backbone construction replaced by a tiny synthetic 12-block ViT: one training epoch, validation, current-run best checkpoint save/reload, test evaluation, plots, CSV and metrics JSON completed. This is a workflow smoke check, not measured VTAB accuracy or pretrained-weight validation.
- Ran the analysis CLI against the generated smoke metrics; it completed.
- All top-level adapter class/function ASTs are unchanged. Split construction code is unchanged; tests verify exact index lists.
- No real dataset, pretrained weight download, GPU run or adapter-correctness evaluation was performed.

Only `data.py`, `single_filter_lora.py`, `train_sfp_lora.py`, and `analyse_results.py` were edited. Added this report and focused regression tests. All other source files and requirements are unchanged. The original uploaded ZIP remains intact.
