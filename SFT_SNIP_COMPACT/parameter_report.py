"""Count actual ViT parameters without downloading pretrained weights or data."""
import argparse
import csv
from pathlib import Path

import timm
from compressed_filters import LowRankFilterBlock, SymbolicFilterBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 11])
    parser.add_argument("--num-classes", type=int, default=37)
    parser.add_argument("--out", default="parameter_report.csv")
    args = parser.parse_args()
    if any(not 1 <= r <= 768 for r in args.ranks) or any(not 1 <= k <= 11 for k in args.blocks):
        parser.error("Require ranks 1..768 and block counts 1..11.")
    model = timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=False,
                              num_classes=args.num_classes)
    original = sum(p.numel() for p in model.parameters())
    width = model.embed_dim
    rows = []
    for count in args.blocks:
        removed = sum(p.numel() for block in model.blocks[:count] for p in block.parameters())
        dense = count * (width * width + width)
        for kind in ("dense", "lowrank", "symbolic"):
            for rank in ([None] if kind == "dense" else args.ranks):
                module = (None if kind == "dense" else LowRankFilterBlock(width, rank) if kind == "lowrank"
                          else SymbolicFilterBlock(width, rank))
                replacement = dense if module is None else count * sum(p.numel() for p in module.parameters())
                total = original - removed + replacement
                rows.append(dict(filter_type=kind, rank=rank, blocks=count, original_params=original,
                                 filter_params=replacement, total_params=total,
                                 filter_reduction_pct=100 * (1 - replacement / dense),
                                 total_reduction_vs_original_pct=100 * (1 - total / original),
                                 total_reduction_vs_dense_same_blocks_pct=100 *
                                 (1 - total / (original - removed + dense))))
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        if row["rank"] in (None, 16):
            print(f"{row['filter_type']:8} blocks={row['blocks']:2} rank={str(row['rank']):4} "
                  f"total={row['total_params']:,} reduction={row['total_reduction_vs_original_pct']:.2f}%")
    print(f"Parameter counts only; accuracy and GPU memory are unmeasured. CSV: {path}")


if __name__ == "__main__":
    main()
