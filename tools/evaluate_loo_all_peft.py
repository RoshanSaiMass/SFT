"""Evaluate a saved all-block PEFT checkpoint with its recorded configuration."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SFT_LOO_ALL_PEFT"))
from train_loo_all_peft import evaluate_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    result = evaluate_summary(args.summary, args.device)
    print(f"Compact replacement: block {result['replaced_block_idx']}")
    print(f"PEFT blocks: {result['peft_block_indices']}")
    print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
    print(f"Test accuracy: {result['final_test_acc']:.2f}%")
    print(f"Updated: {args.summary}")


if __name__ == "__main__":
    main()
