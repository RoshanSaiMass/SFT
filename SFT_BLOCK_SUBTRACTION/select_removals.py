"""Collect candidates, select by mean validation accuracy, then optionally test."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from train_subtraction import evaluate_summary, write_json


def group_id(summary):
    ignored = {"seed", "remove_block", "output_dir", "defer_test", "overwrite",
               "save_misclassified_images", "max_misclassified_images"}
    config = {k: v for k, v in summary["config"].items() if k not in ignored}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def choose(summaries, seeds=None, blocks=None, allow_partial=False):
    """Choose a common removed block per configuration across the specified seeds."""
    groups = defaultdict(list)
    for summary in summaries:
        if summary.get("method") == "drop-one":
            groups[group_id(summary)].append(summary)
    selected = []
    for ident, rows in sorted(groups.items()):
        expected_seeds = set(seeds if seeds is not None else [r["seed"] for r in rows])
        expected_blocks = set(blocks if blocks is not None else range(len(rows[0]["structure"]["remaining_original_indices"]) + 1))
        lookup = {}
        for row in rows:
            key = row["removed_block_idx"], row["seed"]
            if key in lookup:
                raise ValueError(f"Duplicate candidate {key} in configuration {ident}; use a single sweep root.")
            lookup[key] = row
        missing = [(b, s) for b in sorted(expected_blocks) for s in sorted(expected_seeds) if (b, s) not in lookup]
        if missing and not allow_partial:
            raise ValueError(f"Incomplete removal sweep {ident}: missing (block,seed) {missing}. No test selection performed.")
        complete = [b for b in sorted(expected_blocks) if all((b, s) in lookup for s in expected_seeds)]
        if not complete:
            raise ValueError(f"No block has all requested seeds in {ident}.")
        means = {b: statistics.mean(lookup[b, s]["best_val_acc"] for s in expected_seeds) for b in complete}
        winner = max(complete, key=lambda b: (means[b], -b))
        selected.append(dict(group_id=ident, dataset=rows[0]["dataset"], adapter_type=rows[0]["adapter_type"],
                             removed_block_idx=winner, mean_best_val_acc=means[winner],
                             expected_blocks=sorted(expected_blocks), complete_blocks=complete,
                             seeds=sorted(expected_seeds), selection_complete=not missing,
                             runs=[lookup[winner, s] for s in sorted(expected_seeds)]))
    if not selected:
        raise ValueError("No drop-one candidates found.")
    return selected


def flat(summary, path):
    row = dict(summary_path=str(path), method=summary["method"], dataset=summary["dataset"],
               seed=summary["seed"], adapter_type=summary["adapter_type"],
               removed_block=summary["removed_block_idx"], retained_block=summary["retained_block_idx"],
               best_val_acc=summary["best_val_acc"], best_epoch=summary["best_epoch"],
               final_test_acc=summary.get("final_test_acc"), final_test_loss=summary.get("final_test_loss"),
               epochs_trained=summary["epochs_trained"], checkpoint_path=summary["checkpoint_path"],
               checkpoint_bytes=summary["checkpoint_bytes"], total_reduction_pct=summary["total_reduction_pct"],
               config_id=summary["config_id"])
    for section in ("config", "param_breakdown", "efficiency", "structure"):
        for key, value in summary[section].items():
            row[f"{section}.{key}"] = json.dumps(value) if isinstance(value, (list, dict)) else value
    return row


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def collect(root, out_dir=None, seeds=None, blocks=None, evaluate_selected=False, device=None, allow_partial=False):
    root = Path(root); out = Path(out_dir or root / "compiled_results")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest:
        seeds = manifest["seeds"] if seeds is None else seeds
        blocks = manifest["blocks"] if blocks is None else blocks
    summaries, all_rows = [], []
    for path in sorted(root.rglob("metrics_summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("method") not in ("correlation", "drop-one"):
            continue
        if seeds is not None and summary["seed"] not in seeds:
            continue
        if blocks is not None and summary["removed_block_idx"] not in blocks:
            continue
        summary["_summary_path"] = str(path)
        summaries.append(summary); all_rows.append(flat(summary, path))
    if not summaries:
        raise ValueError(f"No subtraction summaries under {root}.")
    write_csv(out / "results_all.csv", all_rows)
    drop_rows = [s for s in summaries if s["method"] == "drop-one"]
    if manifest:
        planned = {group_id({"config": task["config"]}) for task in manifest["tasks"]
                   if task["config"]["seed"] in seeds and task["config"]["remove_block"] in blocks}
        absent = planned - {group_id(row) for row in drop_rows}
        if absent and not allow_partial:
            raise ValueError(f"Incomplete removal sweep: entire planned configurations missing {sorted(absent)}. No test selection performed.")
    if not drop_rows:
        print(f"Collected {len(summaries)} correlation runs; no removal sweep to select.")
        return []
    selected = choose(drop_rows, seeds, blocks, allow_partial)
    best_rows = []
    for winner in selected:
        tests = []
        for run in winner["runs"]:
            if evaluate_selected and not run.get("test_evaluated"):
                updated = evaluate_summary(run["_summary_path"], device)
                run.update(updated)
            if run.get("final_test_acc") is not None:
                tests.append(run["final_test_acc"])
        best_rows.append(dict(group_id=winner["group_id"], dataset=winner["dataset"],
                              adapter_type=winner["adapter_type"], removed_block_idx=winner["removed_block_idx"],
                              mean_best_val_acc=winner["mean_best_val_acc"],
                              mean_test_acc=statistics.mean(tests) if tests else None,
                              std_test_acc=statistics.stdev(tests) if len(tests) > 1 else 0 if tests else None,
                              n_test_seeds=len(tests), seeds=json.dumps(winner["seeds"]),
                              selection_complete=winner["selection_complete"],
                              complete_blocks=json.dumps(winner["complete_blocks"])))
        print(f"{winner['adapter_type']}: selected original block {winner['removed_block_idx']} by mean validation {winner['mean_best_val_acc']:.2f}%")
    write_csv(out / "best_removals.csv", best_rows)
    write_csv(out / "results_all.csv", [flat(s, s["_summary_path"]) for s in summaries])
    write_json(out / "selection.json", [{k: v for k, v in winner.items() if k != "runs"} for winner in selected])
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--blocks", type=int, nargs="+")
    p.add_argument("--evaluate-selected", action="store_true")
    p.add_argument("--device")
    p.add_argument("--allow-partial", action="store_true", help="Explicitly allow selection from incomplete block coverage.")
    args = p.parse_args()
    collect(args.root, args.out_dir, args.seeds, args.blocks, args.evaluate_selected, args.device, args.allow_partial)


if __name__ == "__main__":
    main()
