"""Acquire shared datasets/backbone under locks before exact-count training."""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-dir', default=str(ROOT / 'SFT_NUM_FUNCTION'))
    p.add_argument('--data-dir', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, default=18)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('exact_count_downloads', ROOT / 'slurm/sweep.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    manifest = dict(code_dir=args.code_dir, data_dir=args.data_dir, cache_dir=args.cache_dir)
    helper.configure_environment(manifest)
    helper.check_runtime(manifest)
    helper.prefetch(manifest, dict(dataset=args.dataset, seed=args.seed))


if __name__ == '__main__':
    main()
