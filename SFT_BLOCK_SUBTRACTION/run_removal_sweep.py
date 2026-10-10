"""Fresh one-block deletions across PEFT methods; select with validation only."""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch

import train_subtraction as trainer
from select_removals import collect


def build_manifest(args):
    fields = set(vars(trainer.parser().parse_args([])))
    base = {key: value for key, value in vars(args).items() if key in fields}
    root = Path(args.output_dir).resolve() / f"removal_sweep_{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    tasks = []
    seeds = args.seeds if args.seeds is not None else [args.seed]
    for label, values in (("seeds", seeds), ("blocks", args.blocks), ("adapter types", args.adapter_types)):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {label} would repeat runs.")
    for kind in args.adapter_types:
        for seed in seeds:
            for block in args.blocks:
                task_id = len(tasks)
                folder = root / "tasks" / f"{task_id:04d}_{kind}_seed{seed}_remove{block}"
                config = dict(base, method="drop-one", adapter_type=kind, seed=seed, remove_block=block,
                              pair_index=-1, defer_test=True, output_dir=str(folder / "results"))
                trainer.validate(argparse.Namespace(**config))
                tasks.append(dict(id=task_id, task_dir=str(folder), config=config))
    manifest = dict(root=str(root), tasks=tasks, seeds=seeds, blocks=args.blocks,
                    adapter_types=args.adapter_types, selection_rule="highest mean best validation accuracy across seeds")
    path = root / "manifest.json"
    trainer.write_json(path, manifest)
    return path


def execute(task, resume=False):
    folder = Path(task["task_dir"]); folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "task_status.json"
    if resume:
        candidates = list((folder / "results").rglob("metrics_summary.json"))
        if len(candidates) == 1:
            summary = json.loads(candidates[0].read_text())
            if summary["config_id"] == trainer.configuration_id(task["config"]) and Path(summary["checkpoint_path"]).is_file():
                trainer.write_json(status_path, dict(state="complete", resumed=True, summary_path=str(candidates[0])))
                return True
    trainer.write_json(status_path, dict(state="running", task=task))
    try:
        path = trainer.train(argparse.Namespace(**task["config"]))
        trainer.write_json(status_path, dict(state="complete", summary_path=str(path)))
        return True
    except Exception as error:
        trainer.write_json(status_path, dict(state="failed", error=str(error), traceback=traceback.format_exc()))
        print(f"Task {task['id']} failed: {error}", flush=True)
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    p = trainer.parser()
    p.description = __doc__
    p.add_argument("--adapter-types", nargs="+", choices=trainer.ADAPTERS, default=trainer.ADAPTERS)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--blocks", type=int, nargs="+", choices=range(12), default=list(range(12)))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--manifest")
    p.add_argument("--task-id", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--selection-only", action="store_true", help="Report winner without test evaluation.")
    args = p.parse_args()
    path = Path(args.manifest) if args.manifest else build_manifest(args)
    manifest = json.loads(path.read_text())
    if args.prepare_only:
        print(f"Manifest: {path}\nTasks: {len(manifest['tasks'])}")
        return
    if args.task_id is not None:
        if not 0 <= args.task_id < len(manifest["tasks"]):
            p.error("Task ID outside the manifest.")
        if not execute(manifest["tasks"][args.task_id], args.resume):
            raise SystemExit(1)
        return
    successes = [execute(task, args.resume) for task in manifest["tasks"]]
    # The collector writes the candidate CSV before refusing incomplete selection.
    try:
        collect(manifest["root"], seeds=manifest["seeds"], blocks=manifest["blocks"],
                evaluate_selected=not args.selection_only, device=args.device)
    except ValueError as error:
        print(error, flush=True)
        raise SystemExit(1) from error
    if not all(successes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
