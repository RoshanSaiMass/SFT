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
