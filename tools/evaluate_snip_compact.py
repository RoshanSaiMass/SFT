"""Evaluate a saved SNIP compact checkpoint using its recorded configuration."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SFT_SNIP_COMPACT"))
from train_snip_compact import evaluate_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    result = evaluate_summary(args.summary, args.device)
    print(f"Compact replacement: block {result['replaced_block_idx']}")
    print(f"PEFT: block {result['peft_block_idx']}")
    print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
    print(f"Test accuracy: {result['final_test_acc']:.2f}%")
    print(f"Updated: {args.summary}")


if __name__ == "__main__":
    main()
