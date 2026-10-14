# Sequential SNIP replacement experiments

This copy retains the previously reviewed SNIP candidate search, original-ViT
SNIP ranking for optional targeted PEFT, and dense/low-rank/symbolic filter math.
The candidate schedule discards exactly one lowest whole-network EMA scorer per
epoch, taking 11 rounds to select one of 12 positions. Surviving candidates keep
their trained weights and optimizer moments. The winning filter is reused for
recovery training. `--snip-prune-fraction` is no longer a supported flag.

```bash
python3 run.py snip-compact --dataset pets --filter-type dense --adapter-type none \
  --snip-momentum 0.3 --snip-search-epochs 11 --device cuda
python3 run.py snip-compact --dataset pets --filter-type symbolic --filter-rank 64 \
  --adapter-type dora --lora-rank 32 --lora-alpha 64 --snip-momentum 0.3 --device cuda
```

Run these commands from the repository root. `scheduler/01_snip_only.slurm`
defaults to dense/no-PEFT; `--snip-filter-type symbolic` selects a symbolic-only
comparison. `scheduler/09_snip_compact_peft.slurm` adds original-block targeted
PEFT. Both expose search and compact ranks through the shared scheduler CLI.

The paper describes low-score candidate pruning and reports selection within
11 iterations. This implementation makes the one-removal schedule explicit;
matching every paper protocol choice or selecting identical positions is not
established. Test accuracy and runtime must be measured on the cluster.

## Numerical initialization and scoring update

Dense candidate fitting performs an SVD in float64 but discards singular values
below `max(input_matrix.shape) * eps(captured_input_dtype) * largest_singular_value`.
This avoids treating roundoff in float32 features as float64-quality observations.
The paper does not specify this cutoff. Compact symbolic/low-rank initialization
retains its separate implementation; the dense fix is scoped to this candidate
search and does not modify the original dense SFT backend.

SNIP scoring uses the same `--label-smoothing` as candidate optimization. It still
sums raw weight-times-gradient saliency across the whole network and discards
the lowest EMA. Frozen weights receive scoring derivatives without being updated;
this remains an explicitly disclosed interpretation rather than verified author code.
Candidate gradient clipping now respects `--grad-clip` when enabled.

`snip_search.json` includes `implementation_version=snip_precision_aware_v4`,
`candidate_initialization` for all positions, effective rank, weight norms,
initialization CLS/patch errors, scoring CE/accuracy, and saliency components
(filter, LayerNorm, head, remaining blocks, stem/CLS/position). Those diagnostics
identify why a candidate wins; they do not change the ranking formula.

Start a new campaign/output directory; neither previous results nor candidate
states are overwritten automatically. Matching a published layer index is not
evidence of a successful reproduction by itself.
