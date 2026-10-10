"""Prepare an immutable experiment manifest and execute one SLURM array task."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DATASETS = ["pets", "svhn", "flowers102", "dtd", "caltech101", "cifar100", "fgvc_aircraft",
            "eurosat", "sun397", "pcam", "clevr", "dsprites-loc", "dsprites-ori"]
HERE = Path(__file__).resolve().parent


def prepare(args):
    if args.epochs < 1 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Training budgets must be positive.")
    if any(not 1 <= b <= 11 for b in args.block_counts) or any(not 1 <= r <= 768 for r in args.ranks):
        raise ValueError("Block counts must be 1..11 (retain attention); ranks must be 1..768.")
    for label, values in [("datasets", args.datasets), ("seeds", args.seeds),
                          ("block counts", args.block_counts), ("ranks", args.ranks)]:
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {label} would repeat experiments.")
    root = Path(args.output_root or HERE / "outputs" / f"sweep_{time.time_ns()}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Existing manifest will not be overwritten: {manifest_path}")
    (root / "logs").mkdir(exist_ok=True)
    tasks = []
    budgets = ["fixed", "early"] if args.training_budget == "both" else [args.training_budget]
    configurations = [("dense", None)] + [(kind, rank) for kind in ("lowrank", "symbolic") for rank in args.ranks]
    for dataset in args.datasets:
        for seed in args.seeds:
            for budget in budgets:
                for blocks in args.block_counts:
                    for kind, rank in configurations:
                        task_id = len(tasks)
                        name = f"{task_id:05d}_{dataset}_{kind}_r{rank or 0}_blocks{blocks}_{budget}_seed{seed}"
                        task_dir = root / "tasks" / name
                        command = ["--dataset", dataset, "--mode", "sft", "--seed", str(seed),
                                   "--filter-type", kind, "--num-filter-blocks", str(blocks),
                                   "--data-dir", str(Path(args.data_dir or HERE / "data").resolve()),
                                   "--output-dir", str(task_dir / "results"),
                                   "--save-misclassified-images", "false", "--device", "cuda",
                                   "--epochs", str(args.epochs if budget == "fixed" else -1),
                                   "--max-epochs", str(args.max_epochs), "--patience", str(args.patience)]
                        if rank is not None:
                            command += ["--filter-rank", str(rank)]
                        tasks.append(dict(id=task_id, dataset=dataset, seed=seed, budget=budget,
                                          filter_type=kind, rank=rank, blocks=blocks,
                                          task_dir=str(task_dir), args=command))
    manifest_path.write_text(json.dumps(dict(code_dir=str(HERE), root=str(root), tasks=tasks), indent=2))
    print(manifest_path)


def run(args):
    manifest = json.loads(Path(args.manifest).read_text())
    if not 0 <= args.task_id < len(manifest["tasks"]):
        raise ValueError("Task ID is outside the manifest.")
    task = manifest["tasks"][args.task_id]
    folder = Path(task["task_dir"]); folder.mkdir(parents=True, exist_ok=True)
    status = dict(task_id=args.task_id, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  state="preparing", task=task)
    path = folder / "task_status.json"
    path.write_text(json.dumps(status, indent=2))
    exit_code = 1
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the selected environment; GPU task refused.")
        command = [sys.executable, str(Path(manifest["code_dir"]) / "train_sfp_lora.py"), *task["args"]]
        print("Launching:", command, flush=True)
        status.update(state="running", command=command)
        path.write_text(json.dumps(status, indent=2))
        exit_code = subprocess.run(command, cwd=manifest["code_dir"], check=False).returncode
        summaries = list((folder / "results").rglob("metrics_summary.json"))
        status["state"] = "complete" if exit_code == 0 and len(summaries) == 1 else "failed"
        if status["state"] == "failed" and exit_code == 0:
            exit_code = 1
        status["summary_paths"] = [str(p) for p in summaries]
    except Exception as error:
        status.update(state="failed", error=str(error))
        print(f"Task failed: {error}", file=sys.stderr)
    finally:
        status.update(exit_code=exit_code,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        path.write_text(json.dumps(status, indent=2))
    return exit_code


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--datasets", nargs="+", choices=DATASETS, default=["pets"])
    prep.add_argument("--seeds", nargs="+", type=int, default=[18])
    prep.add_argument("--block-counts", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    prep.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
    prep.add_argument("--training-budget", choices=["fixed", "early", "both"], default="fixed")
    prep.add_argument("--epochs", type=int, default=100)
    prep.add_argument("--max-epochs", type=int, default=200)
    prep.add_argument("--patience", type=int, default=10)
    prep.add_argument("--data-dir")
    prep.add_argument("--output-root")
    worker = sub.add_parser("run")
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    try:
        return prepare(args) if args.command == "prepare" else run(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
