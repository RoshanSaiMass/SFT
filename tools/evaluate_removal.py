"""Evaluate one existing subtraction checkpoint using its saved configuration."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SFT_BLOCK_SUBTRACTION"))
from train_subtraction import evaluate_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True, help="metrics_summary.json beside the saved checkpoint")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    result = evaluate_summary(args.summary, device=args.device)
    print(f"Removed original block: {result['removed_block_idx']}")
    print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
    print(f"Test accuracy: {result['final_test_acc']:.2f}%")
    print(f"Updated: {args.summary}")


if __name__ == "__main__":
    main()
