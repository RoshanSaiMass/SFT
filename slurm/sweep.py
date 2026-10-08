#!/usr/bin/env python3
"""Prepare, run and collect the dataset × method × seed SLURM array.

Preparation/collection use only the standard library (Excel output is optional).
Training tasks require the already-installed project dependencies and CUDA.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

DATASETS = ["pets", "svhn", "flowers102", "dtd", "caltech101", "cifar100",
            "fgvc_aircraft", "eurosat", "sun397", "pcam", "clevr",
            "dsprites-loc", "dsprites-ori"]
BACKBONE = "vit_base_patch16_224.augreg_in21k"


def method_configs():
    configs = [("sft", ["--mode", "sft"]),
               ("full_finetune", ["--mode", "full_finetune"])]
    ortho = ["--lora-ortho-lambda1", "0.0001", "--lora-ortho-lambda2", "0.0001"]
    for family in ("lora", "dora"):
        base = ["--adapter-type", family, "--lora-rank", "32", "--lora-alpha", "32"]
        configs.extend([(family, base),
                        (family + "_loftq", base + ["--init-method", "loftq"]),
                        (family + "_ortho", base + ortho)])
    for family in ("paca", "rpaca"):
        base = ["--adapter-type", family, "--paca-rank", "32", "--lora-alpha", "32"]
        configs.append((family + "_direct", base + ["--paca-tuner", "direct"]))
        for tuner in ("lora", "dora"):
            fused = base + ["--paca-tuner", tuner, "--paca-adapter-rank", "4"]
            configs.extend([(family + "_" + tuner, fused),
                            (family + "_" + tuner + "_ortho", fused + ortho)])
    for family in ("unilora", "unidora"):
        configs.append((family, ["--adapter-type", family, "--lora-rank", "4",
                                 "--lora-alpha", "32", "--unilora-dim", "72000"]))
    return [{"config_id": name, "method_args": args} for name, args in configs]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare(code_dir, data_dir, cache_dir, output_root, seeds=(18,), epochs=100,
            max_epochs=200, patience=10, num_samples=1000, batch_size=32,
            save_misclassified=False, training_budget="fixed"):
    code = Path(code_dir).expanduser().resolve()
    if not (code / "train_sfp_lora.py").is_file():
        raise ValueError(f"Training source missing: {code / 'train_sfp_lora.py'}")
    if epochs <= 0:
        raise ValueError("epochs must be positive; use training_budget='early' for early stopping")
    if training_budget not in ("fixed", "early", "both"):
        raise ValueError("training_budget must be fixed, early, or both")
    if min(max_epochs, patience, batch_size) <= 0 or num_samples < 80:
        raise ValueError("Positive budget/batch settings and at least 80 samples are required")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("Use distinct integer seeds in [0, 2**32)")
    settings = dict(epochs=epochs, max_epochs=max_epochs, patience=patience,
                    num_samples=num_samples, batch_size=batch_size,
                    save_misclassified=save_misclassified, training_budget=training_budget)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_root).expanduser().resolve() / f"sweep_{stamp}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    tasks = []
    for dataset in DATASETS:
        for method in method_configs():
            for seed in seeds:
                budgets = ("fixed", "early") if training_budget == "both" else (training_budget,)
                for budget in budgets:
                    index = len(tasks)
                    task_dir = root / "tasks" / f"{index:04d}_{dataset}_{method['config_id']}_{budget}_seed{seed}"
                    common = ["--dataset", dataset, "--seed", str(seed), "--device", "cuda",
                              "--epochs", str(epochs if budget == "fixed" else -1),
                              "--max-epochs", str(max_epochs), "--patience", str(patience),
                              "--num-samples", str(num_samples), "--batch-size", str(batch_size),
                              "--save-misclassified-images", str(save_misclassified).lower(),
                              "--output-dir", str(task_dir / "results")]
                    tasks.append(dict(task_id=index, dataset=dataset, seed=seed, budget=budget,
                                      **method, task_dir=str(task_dir),
                                      train_args=common + method["method_args"]))
    manifest = dict(schema_version=1, created_utc=stamp, sweep_root=str(root),
                    code_dir=str(code), data_dir=str(Path(data_dir).expanduser().resolve()),
                    cache_dir=str(Path(cache_dir).expanduser().resolve()),
                    settings=settings, seeds=list(seeds), tasks=tasks)
    write_json(root / "manifest.json", manifest)
    write_csv(root / "manifest.csv", [dict(task_id=t["task_id"], dataset=t["dataset"],
              config_id=t["config_id"], budget=t["budget"], seed=t["seed"], task_dir=t["task_dir"],
              method_args=json.dumps(t["method_args"]), train_args=json.dumps(t["train_args"]),
              **settings) for t in tasks])
    return root


def configure_environment(manifest):
    cache = Path(manifest["cache_dir"])
    values = {"HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "huggingface" / "hub",
              "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub", "TORCH_HOME": cache / "torch",
              "XDG_CACHE_HOME": cache / "xdg", "MPLCONFIGDIR": cache / "matplotlib"}
    for key, directory in values.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def check_runtime(manifest):
    code = Path(manifest["code_dir"])
    for name in ("train_sfp_lora.py", "data.py", "single_filter_lora.py", "snip_selection.py"):
        if not (code / name).is_file():
            raise RuntimeError(f"Required source file missing: {code / name}")
    sys.path.insert(0, str(code))
    for name in ("torch", "torchvision", "timm", "numpy", "matplotlib", "pandas",
                 "openpyxl", "scipy", "h5py", "gdown"):
        importlib.import_module(name)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this job; refusing a CPU training fallback")
    print(f"CUDA: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)


def locked_prepare(lock_path, marker_path, callback):
    """Only publish a ready marker after the entire preparation succeeds."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker_path.exists():
            return
        callback()
        marker_path.write_text(datetime.now(timezone.utc).isoformat() + "\n")


def prefetch(manifest, task):
    from data import get_dataset_by_name
    import timm
    data = Path(manifest["data_dir"])
    data.mkdir(parents=True, exist_ok=True)
    # Both dSprites tasks share one archive; serialize them with the same lock.
    dataset_key = "dsprites" if task["dataset"].startswith("dsprites-") else task["dataset"]
    source_hash = hashlib.sha256((Path(manifest["code_dir"]) / "data.py").read_bytes()).hexdigest()[:12]
    lock_dir = data / ".sft_download_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset():
        for training in (True, False):
            dataset = get_dataset_by_name(task["dataset"], root=str(data), train=training,
                                          transform=None, seed=task["seed"])
            del dataset
            gc.collect()

    print(f"Preparing shared dataset {task['dataset']} (locked)", flush=True)
    locked_prepare(lock_dir / f"{dataset_key}.lock",
                   lock_dir / f"{dataset_key}_{source_hash}.ready", download_dataset)
    cache = Path(manifest["cache_dir"])

    def download_backbone():
        # CPU-only acquisition, released before the real GPU model is constructed.
        model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
        del model
        gc.collect()

    print("Preparing shared pretrained backbone (locked)", flush=True)
    locked_prepare(cache / "backbone.lock", cache / "backbone.ready", download_backbone)


def execute_training(manifest, task):
    task_dir = Path(task["task_dir"])
    command = [sys.executable, str(Path(manifest["code_dir"]) / "train_sfp_lora.py"),
               *task["train_args"]]
    write_json(task_dir / "status.json", dict(status="running", command=command))
    print("Training argv: " + json.dumps(command), flush=True)
    result = subprocess.run(command, cwd=task_dir)
    summaries = list((task_dir / "results").glob("*/metrics_summary.json"))
    returncode = result.returncode if result.returncode else (0 if len(summaries) == 1 else 1)
    write_json(task_dir / "status.json", dict(status="complete" if returncode == 0 else "failed",
               returncode=returncode, training_returncode=result.returncode, command=command,
               summary_path=str(summaries[0]) if len(summaries) == 1 else None))
    return returncode if returncode >= 0 else 128 - returncode


def run_task(manifest, task_id):
    if task_id < 0 or task_id >= len(manifest["tasks"]):
        raise ValueError(f"Task index {task_id} is outside this manifest")
    task = manifest["tasks"][task_id]
    task_dir = Path(task["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(task_dir / "status.json", dict(status="preparing"))
    try:
        data = Path(manifest["data_dir"])
        data.mkdir(parents=True, exist_ok=True)
        link = task_dir / "data"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(data, target_is_directory=True)
        if not link.is_symlink() or link.resolve() != data.resolve():
            raise RuntimeError(f"Task data link points to the wrong location: {link}")
        configure_environment(manifest)
        check_runtime(manifest)
        prefetch(manifest, task)
        return execute_training(manifest, task)
    except Exception as error:
        write_json(task_dir / "status.json", dict(status="failed", returncode=1,
                                                  error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        collect_locked(manifest)


def collect(manifest):
    rows = []
    for task in manifest["tasks"]:
        task_dir = Path(task["task_dir"])
        status_path = task_dir / "status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {"status": "waiting"}
        row = dict(task_id=task["task_id"], dataset=task["dataset"], config_id=task["config_id"],
                   budget=task["budget"],
                   seed=task["seed"], status=state["status"], returncode=state.get("returncode"),
                   error=state.get("error"), task_dir=str(task_dir),
                   method_args=json.dumps(task["method_args"]))
        summaries = list((task_dir / "results").glob("*/metrics_summary.json"))
        if state["status"] == "complete" and len(summaries) != 1:
            row.update(status="failed", error="Completed task has no unique metrics_summary.json")
        if len(summaries) == 1 and state["status"] in ("complete", "failed"):
            try:
                summary = json.loads(summaries[0].read_text())
            except (OSError, ValueError) as error:
                summary = {}
                row.update(status="failed", error=f"Unreadable metrics summary: {error}")
            for key in ("mode", "best_val_acc", "best_epoch", "final_test_acc", "final_test_loss",
                        "epochs_trained", "pruned_block_idx", "lora_rank", "paca_columns",
                        "paca_adapter_rank", "unilora_dim"):
                row["resolved_mode" if key == "mode" else key] = summary.get(key)
            for key in ("trainable_params", "total_params", "trainable_pct", "train_compute_seconds",
                        "steady_state_throughput_samples_per_sec", "peak_gpu_mem_allocated_mb",
                        "peak_gpu_mem_reserved_mb"):
                row[key] = summary.get("efficiency", {}).get(key)
            row["summary_path"] = str(summaries[0])
        rows.append(row)
    root = Path(manifest["sweep_root"])
    counts = {status: sum(row["status"] == status for row in rows)
              for status in sorted({row["status"] for row in rows})}
    groups = {}
    for row in rows:
        if row["status"] == "complete":
            key = (row["dataset"], row["config_id"], row["budget"])
            groups.setdefault(key, []).append(row)
    grouped_stats = []
    for (dataset, config_id, budget), members in groups.items():
        entry = dict(dataset=dataset, config_id=config_id, budget=budget,
                     n_runs=len(members), seeds=[row["seed"] for row in members])
        for metric in ("best_val_acc", "final_test_acc"):
            values = [row[metric] for row in members if row.get(metric) is not None]
            entry[metric + "_mean"] = statistics.mean(values) if values else None
            entry[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        grouped_stats.append(entry)
    write_json(root / "aggregate_summary.json", dict(status_counts=counts, runs=rows,
                                                     per_group_stats=grouped_stats))
    write_csv(root / "aggregate_summary.csv", rows)
    write_csv(root / "aggregate_stats.csv", grouped_stats)
    try:
        import pandas as pd
        with pd.ExcelWriter(root / "aggregate_summary.xlsx", engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="runs", index=False)
            pd.DataFrame(grouped_stats).to_excel(writer, sheet_name="per_group_stats", index=False)
    except ImportError:
        pass
    print(json.dumps(counts, sort_keys=True))
    print(f"Results: {root / 'aggregate_summary.csv'}")
    return rows


def collect_locked(manifest):
    with (Path(manifest["sweep_root"]) / "collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return collect(manifest)


def record_failure(manifest, task_id, returncode, error):
    task_dir = Path(manifest["tasks"][task_id]["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(task_dir / "status.json", dict(status="failed", returncode=returncode, error=error))
    collect_locked(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for key in ("code-dir", "data-dir", "cache-dir", "output-root"):
        prep.add_argument("--" + key, required=True)
    prep.add_argument("--seeds", nargs="+", type=int, default=[18])
    prep.add_argument("--training-budget", choices=["fixed", "early", "both"], default="fixed")
    for key, default in (("epochs", 100), ("max-epochs", 200), ("patience", 10),
                         ("num-samples", 1000), ("batch-size", 32)):
        prep.add_argument("--" + key, type=int, default=default)
    prep.add_argument("--save-misclassified", action="store_true")
    for action in ("task", "collect", "record-failure"):
        sub = commands.add_parser(action)
        sub.add_argument("--manifest", required=True)
        if action in ("task", "record-failure"):
            sub.add_argument("--task-id", required=True, type=int)
        if action == "record-failure":
            sub.add_argument("--returncode", required=True, type=int)
            sub.add_argument("--error", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        print(prepare(**args))
        return 0
    manifest = json.loads(Path(args["manifest"]).read_text())
    if command == "task":
        return run_task(manifest, args["task_id"])
    if command == "record-failure":
        record_failure(manifest, args["task_id"], args["returncode"], args["error"])
        return 0
    rows = collect_locked(manifest)
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
