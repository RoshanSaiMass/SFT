# Stage-two PEFT integration audit

This audit checks the code supplied by the user, building on the earlier SFP corrections. It checks implementation/integration behavior against explicit matrix formulas and independent invariants, not author-repository or full benchmark parity. The only research paper supplied was the SFP paper; the individual PEFT papers/repositories have not been independently audited here.

## Placement verdict

LoRA, DoRA, direct PaCA/RPaCA, fused PaCA/RPaCA with LoRA or DoRA tuners, Uni-LoRA, and Uni-DoRA are inserted in qkv, attention proj, fc1 and fc2 projections of original transformer blocks. Filter-substituted blocks are excluded. Patch embedding, positional/CLS embeddings and frozen base projections stay outside the adapted set. The SFP filters, remaining LayerNorms and head train alongside the selected adaptation parameters.

The CLI exposes SFP-plus-adapter hybrids, not standalone PEFT on an entirely unmodified backbone. Full fine-tuning and plain SFP are separate presets. Fused column adaptation uses --adapter-type paca/rpaca plus --paca-tuner lora/dora; it does not use a separate --adapter-type doca.

## Confirmed issues corrected

| Issue | Correction |
|---|---|
| Standalone DoRA magnitude was frozen and absent from the adapter optimizer group | Identify adapter parameters by module ownership, including magnitude and the single Uni-LoRA bank parameter; reuse that identity set for freezing, grouping and reporting. Keep checkpoint parameter names unchanged. |
| PaCA weights stored as buffers were absent from component counts; direct PaCA was labeled trainable backbone | Include logical projection weights/biases and classify adaptation parameters correctly. Count selected direct PaCA columns once. Reuse the same counts for efficiency and component reports. These are logical model counts, not physical allocated memory. |
| Parameter pie included total/trainable summary values as overlapping categories | Plot only component categories. |
| Diagnostic reconstructed every checkpoint as plain LoRA | Reconstruct family, column count, tuner, adapter rank, shared dimension, backbone and filter options from saved metadata. Do not rerun LoftQ on a saved quantized checkpoint. Support relocated local checkpoint paths and weights-only loading. |
| Diagnostic used BA alone for DoRA and could not inspect other families | Report effective DoRA/Uni-DoRA weight deltas, PaCA column deltas, fused-column deltas and Uni-LoRA deltas. RPaCA's delta is relative to the current saved base; earlier committed epochs are not recoverable as a total delta without original weights. Remove unsupported causal claims that small deltas prove undertraining. |
| Preset values were resolved after compatibility validation | Resolve presets first; incompatible orthogonality settings for direct PaCA/Uni adapters are rejected before dataset acquisition. Plain SFP rejects adapter-family combinations that would silently retain adapters. |
| Rank compensation injected adapters twice and could quantize twice under LoftQ | Calculate the compensated rank before injection; initialize exactly once at the final rank. Reject multi-filter compensation because that path is not implemented. |
| DoRA input dropout perturbed the frozen projection; Uni-DoRA ignored dropout | Apply dropout to low-rank input only. Preserve the deterministic base projection. Default dropout=0 remains mathematically equivalent to the previous effective-weight equation. |
| Invalid LoftQ ranks/scaling could fail inside an SVD/copy operation | Validate SVD rank against matrix dimensions and require positive alpha, bits and iteration count. |
| DoRA summaries said sft_lora; fused LoRA and fused DoRA could share an analysis label | Save the actual adapter family and distinguish fused tuners in analysis labels. |
| Error-image outputs could overwrite another replacement-position/configuration run | Store error images inside the unique run directory. Their paths are recorded in metrics_summary.json. The error-diff helper accepts relative run-directory paths. |

New metadata records the backbone and unused Uni-LoRA projection slots. The random shared-vector projection may have unused coordinates; normalization applies to occupied coordinates. Parameter counts still describe registered parameters.

## Validation actually performed

Command:

```bash
python -m pytest -q tests
```

Result: **118 passed** on CPU (PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, timm 1.0.30).

Coverage:

- Twelve configurations: LoRA and DoRA with default and LoftQ-style initialization; direct PaCA/RPaCA; fused PaCA/RPaCA with each tuner; Uni-LoRA/Uni-DoRA.
- Each configuration with one and two filter replacements: exact target placement, trainable manifest, unchanged frozen parameters, useful gradient/parameter updates and state-dict round trips.
- Actual trainer two-epoch runs using a small 3-block synthetic ViT and synthetic 64-image training set, including validation, best-checkpoint restoration, test evaluation, history CSV and metrics JSON. Dataset acquisition and backbone construction were replaced for these smoke checks; expensive plots were stubbed. These are not benchmark accuracy runs.
- Actual full-finetuning/plain-SFP presets and selective-orthogonality training configurations.
- Explicit effective-weight formulas, zero-initialization output preservation, column-confined fused updates, resampling output continuity and optimizer-state clearing.
- Every adapter family's saved-metadata diagnostic reconstruction and predictions.
- Shared Uni vector registered once, shared correctly across layers and preserved by deepcopy.
- Compensated LoftQ called only once per target projection.
- Dropout and invalid-option behavior.
- Existing stage-one tests for SFP initialization, split preservation and evaluation consistency.

All Python source parses; the training --help command now works. Data/split code and SNIP source are byte-identical to the stage-one version. No real dataset, pretrained-weight download, GPU test, production-scale memory/timing test or benchmark accuracy result was performed.

## Remaining qualifications

- LoRA/DoRA forward and gradient integration are tested against the implemented equations. Exact upstream-library parity, including normalization-gradient conventions, has not been established.
- LoftQ remains uniform quantize/dequantize plus SVD initialization, not packed NF4/QLoRA and not a claim of quantized-memory savings.
- Direct PaCA rebuilds a full floating-point weight in forward; it does not implement the original method's optimized memory-saving kernel.
- Fused column methods and SFP-plus-PEFT combinations are extensions. Correct integration does not establish their superiority.
- The SFP candidate-search gap and training protocol differences remain unchanged. No all-block plotting/sweep feature was added in this audit.
- Use consistent seeds, replacement positions, preprocessing, training budgets and explicit parameter-budget reporting across method comparisons. Analysis labels now distinguish fused tuners, but rank/column/subspace sweeps still need explicit grouping/filtering to avoid pooling different capacities.
- Existing old DoRA runs did not train magnitude correctly. Rerun those for comparison with this corrected version.

## Example invocation patterns

Append the same dataset, seed, epoch and replacement-position settings to each command. To use SNIP, omit --pruned-block; for a fixed-position comparison, supply it consistently.

```bash
python train_sfp_lora.py --mode sft
python train_sfp_lora.py --adapter-type lora --lora-rank 16
python train_sfp_lora.py --mode sft_lora_ortho --lora-rank 16
python train_sfp_lora.py --adapter-type dora --lora-rank 16
python train_sfp_lora.py --adapter-type paca --paca-rank 16
python train_sfp_lora.py --adapter-type rpaca --paca-rank 16
python train_sfp_lora.py --adapter-type paca --paca-tuner lora --paca-rank 32 --paca-adapter-rank 4
python train_sfp_lora.py --adapter-type paca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 4
python train_sfp_lora.py --adapter-type rpaca --paca-tuner lora --paca-rank 32 --paca-adapter-rank 4
python train_sfp_lora.py --adapter-type rpaca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 4
python train_sfp_lora.py --adapter-type unilora --lora-rank 4 --unilora-dim 72000
python train_sfp_lora.py --adapter-type unidora --lora-rank 4 --unilora-dim 72000
```

Error-image paths now live below each run directory, rather than directly below the global output root. Use the path recorded in metrics_summary.json when invoking analyze_misclassified_diff.py.

Changed source files in this stage: single_filter_lora.py, train_sfp_lora.py, lora_delta_diagnostic.py, analyse_results.py, plotting.py. Added tests/test_peft_integration.py and tests/test_peft_training.py. The existing split and SNIP behavior remain unchanged.
