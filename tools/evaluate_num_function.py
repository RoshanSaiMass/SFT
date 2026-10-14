"""Evaluate an exact-count symbolic LOO checkpoint with its saved configuration."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SFT_NUM_FUNCTION"))
from train_num_function import evaluate_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    result = evaluate_summary(args.summary, args.device)
    print(f"Exact function count: {result['num_functions']}")
    print(f"Selected functions: {result['selected_terms']}")
    print(f"Compact replacement: block {result['replaced_block_idx']}")
    print(f"PEFT: block {result['peft_block_idx']}")
    print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
    print(f"Test accuracy: {result['final_test_acc']:.2f}%")


if __name__ == "__main__":
    main()
