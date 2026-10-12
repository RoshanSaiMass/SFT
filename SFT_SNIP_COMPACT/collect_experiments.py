"""One CSV row per completed experiment; preserves individual block/config choices."""
import argparse
import csv
import json
from pathlib import Path


def collect(root):
    rows = []
    for path in sorted(Path(root).rglob("metrics_summary.json")):
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f"Skipping unreadable summary {path}: {error}"); continue
        row = {"summary_path": str(path)}
        for key, value in summary.items():
            if key in ("compression", "efficiency", "param_breakdown") and isinstance(value, dict):
                row.update({f"{key}.{k}": json.dumps(v) if isinstance(v, (list, dict)) else v
                            for k, v in value.items()})
            else:
                row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--out", default="compiled_results/experiments_all.csv")
    args = parser.parse_args()
    rows = collect(args.root)
    if not rows:
        parser.exit(1, f"No readable completed summaries under {args.root}\n")
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} completed runs to {path}. Missing/failed runs have no summary row.")


if __name__ == "__main__":
    main()
