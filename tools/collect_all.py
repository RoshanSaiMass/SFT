"""Compile every saved run without retraining, reranking, or evaluating test."""
import argparse
import csv
import json
from pathlib import Path


def flatten(value, prefix=""):
    row = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            row.update(flatten(item, name))
        else:
            row[name] = json.dumps(item) if isinstance(item, list) else item
    return row


def collect(root, output):
    rows = []
    for path in sorted(Path(root).rglob("metrics_summary.json")):
        summary = json.loads(path.read_text())
        row = flatten(summary)
        if summary.get("method") in ("drop-one", "correlation"):
            workflow = summary["method"]
            blocks = [summary["removed_block_idx"]]
        elif summary.get("method") == "snip-compact":
            workflow = "snip-compact"
            blocks = summary["pruned_block_idx"]
        else:
            workflow = summary.get("filter_type", "original")
            blocks = summary.get("pruned_block_idx", [])
        row.update(summary_path=str(path.resolve()), workflow=workflow,
                   block_indices=json.dumps(blocks),
                   test_available=summary.get("final_test_acc") is not None)
        rows.append(row)
    if not rows:
        raise ValueError(f"No saved metrics_summary.json files under {root}")
    first = ["workflow", "dataset", "seed", "adapter_type", "mode", "block_indices",
             "best_val_acc", "final_test_acc", "test_available", "summary_path"]
    fields = first + sorted({k for row in rows for k in row} - set(first))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(output)
    print(f"Wrote {len(rows)} saved runs to {output}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    try:
        collect(args.root, args.out)
    except (OSError, ValueError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
