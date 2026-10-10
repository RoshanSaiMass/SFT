#!/bin/bash
# Compatible with the original SFT-sft-experiment.zip: invokes experiment.sbatch
# and run_grid.py unchanged. Use bash, not sbatch, to prepare logs/manifest first.
set -euo pipefail
CODE_DIR="${CODE_DIR:-/export/home/achyut/Sarvesh/SFT_FINAL/SFT_EXPERIEMTN}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_CONCURRENT must be a positive integer.\n' >&2
    exit 2
fi
manifest="$(python3 - "$CODE_DIR" "$@" <<'PY'
import argparse
import json
from pathlib import Path
import sys
import time

code = Path(sys.argv[1]).resolve()
parser = argparse.ArgumentParser(description="Submit all block positions, or a multi-block count sweep.")
parser.add_argument("--datasets", nargs="+", default=["pets"],
                    choices=["pets", "svhn", "flowers102", "dtd", "caltech101", "cifar100",
                             "fgvc_aircraft", "eurosat", "sun397", "pcam", "clevr",
                             "dsprites-loc", "dsprites-ori"])
parser.add_argument("--seeds", nargs="+", type=int, default=[18])
parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
choice = parser.add_mutually_exclusive_group()
choice.add_argument("--block-indices", nargs="+", type=int, help="Individual positions (default: 0..11).")
choice.add_argument("--block-counts", nargs="+", type=int, help="Numbers of SNIP-selected blocks to replace, 1..11.")
parser.add_argument("--training-budget", choices=["fixed", "early", "both"], default="fixed")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--max-epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--data-dir")
parser.add_argument("--output-root")
args = parser.parse_args(sys.argv[2:])
if not all((code / name).is_file() for name in ("experiment.sbatch", "run_grid.py", "train_sfp_lora.py")):
    parser.error(f"Experiment code not found at {code}; set CODE_DIR to your extracted folder.")
positions = args.block_counts is None
blocks = args.block_indices if args.block_indices is not None else list(range(12))
if not positions:
    blocks = args.block_counts
if any(not (0 <= b <= 11 if positions else 1 <= b <= 11) for b in blocks):
    parser.error("Positions must be 0..11; replacement counts must be 1..11 (retain attention).")
if any(not 1 <= rank <= 768 for rank in args.ranks):
    parser.error("Ranks must be 1..768.")
if min(args.epochs, args.max_epochs, args.patience) < 1:
    parser.error("Epoch budgets and patience must be positive.")
if any(not 0 <= seed < 2**32 for seed in args.seeds):
    parser.error("Seeds must be 0..2^32-1.")
for label, values in (("datasets", args.datasets), ("seeds", args.seeds),
                      ("ranks", args.ranks), ("blocks", blocks)):
    if len(set(values)) != len(values):
        parser.error(f"Duplicate {label} would repeat jobs.")
root = Path(args.output_root or code / "outputs" / f"block_sweep_{time.time_ns()}").resolve()
path = root / "manifest.json"
if path.exists():
    parser.error(f"Existing manifest will not be overwritten: {path}")
root.mkdir(parents=True, exist_ok=True)
(root / "logs").mkdir(exist_ok=True)
data = Path(args.data_dir or code.parent / "data").resolve()
configurations = [("dense", None)] + [(kind, rank) for kind in ("lowrank", "symbolic") for rank in args.ranks]
budgets = ["fixed", "early"] if args.training_budget == "both" else [args.training_budget]
tasks = []
for dataset in args.datasets:
    for seed in args.seeds:
        for budget in budgets:
            for block in blocks:
                for kind, rank in configurations:
                    task_id = len(tasks)
                    label = f"block{block}" if positions else f"count{block}"
                    folder = root / "tasks" / f"{task_id:05d}_{dataset}_{kind}_r{rank or 0}_{label}_{budget}_seed{seed}"
                    command = ["--dataset", dataset, "--mode", "sft", "--seed", str(seed),
                               "--filter-type", kind, "--data-dir", str(data),
                               "--output-dir", str(folder / "results"), "--device", "cuda",
                               "--save-misclassified-images", "false", "--epochs",
                               str(args.epochs if budget == "fixed" else -1),
                               "--max-epochs", str(args.max_epochs), "--patience", str(args.patience)]
                    command += ["--pruned-block" if positions else "--num-filter-blocks", str(block)]
                    if rank is not None:
                        command += ["--filter-rank", str(rank)]
                    tasks.append(dict(id=task_id, dataset=dataset, seed=seed, budget=budget,
                                      filter_type=kind, rank=rank, blocks=1 if positions else block,
                                      block_index=block if positions else None,
                                      task_dir=str(folder), args=command))
path.write_text(json.dumps(dict(code_dir=str(code), root=str(root), sweep_type="positions" if positions else "counts",
                               tasks=tasks), indent=2))
print(path)
PY
)"
# argparse --help printed inside command substitution; display it without submitting.
if [[ "$manifest" == usage:* ]]; then
    printf '%s\n' "$manifest"
    exit 0
fi
task_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$manifest")"
sweep_root="$(dirname "$manifest")"
printf 'Manifest: %s\nTasks: %s\nResults: %s/tasks\n' "$manifest" "$task_count" "$sweep_root"
sbatch --job-name=sft_exp_blocks --array="0-$((task_count-1))%$MAX_CONCURRENT" \
    --output="$sweep_root/logs/out_%x_%A_%a.log" \
    --error="$sweep_root/logs/err_%x_%A_%a.log" \
    "$CODE_DIR/experiment.sbatch" "$CODE_DIR" "$manifest"
