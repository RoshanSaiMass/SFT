"""Measure the actual ViT-B/16 trainable budget without downloads or a GPU."""
import argparse
import gc
import json
from pathlib import Path
import sys
import torch
import timm
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'SFT_LOO_BOTTLENECK'))
import train_bottleneck as train
from loo_selection import assert_trainable_scope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-classes', type=int, default=37)
    p.add_argument('--output')
    args = p.parse_args()
    if args.num_classes < 1:
        p.error('num-classes must be positive')
    torch.set_num_threads(2)
    rows = []
    for width in (64, 96, 128):
        for rank in (8, 16):
            for adapter in ('lora', 'dora'):
                model = timm.create_model(train.BACKBONE, pretrained=False, num_classes=args.num_classes)
                original = sum(p.numel() for p in model.parameters())
                cfg = train.parser().parse_args(['--filter-rank', str(width), '--lora-rank', str(rank),
                                                '--adapter-type', adapter])
                train.install_compact(model, cfg, 3)
                train.apply_top_peft(model, cfg, 3, 0)
                assert_trainable_scope(model, 3, 0)
                count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                row = dict(num_classes=args.num_classes, filter_rank=width, peft_rank=rank,
                           adapter=adapter, original_model_params=original,
                           trainable_params=count, trainable_pct_original=100*count/original,
                           stored_model_params=sum(p.numel() for p in model.parameters()))
                rows.append(row)
                print(json.dumps(row), flush=True)
                del model
                gc.collect()
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
