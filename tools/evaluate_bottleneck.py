"""Evaluate an nonlinear bottleneck LOO checkpoint with its saved configuration."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SFT_LOO_BOTTLENECK"))
from train_bottleneck import evaluate_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    result = evaluate_summary(args.summary, args.device)
    print(f"Bottleneck width: {result['bottleneck_rank']}")
    print(f"Compact replacement: block {result['replaced_block_idx']}")
    print(f"PEFT: block {result['peft_block_idx']}")
    print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
    print(f"Test accuracy: {result['final_test_acc']:.2f}%")


if __name__ == "__main__":
    main()
