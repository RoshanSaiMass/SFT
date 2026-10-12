"""Evaluate a saved sequential SNIP checkpoint from its recorded configuration."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'SFT_SNIP_COMPACT'))
from train_snip_compact import evaluate_summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = evaluate_summary(args.summary, args.device)
    print(f"Test accuracy: {result['final_test_acc']:.2f}%")

if __name__ == '__main__':
    main()
