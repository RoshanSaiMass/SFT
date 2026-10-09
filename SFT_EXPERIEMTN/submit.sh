#!/bin/bash
set -euo pipefail
CODE_DIR="${CODE_DIR:-/export/home/achyut/Sarvesh/SFT_FINAL/SFT_EXPERIEMTN}"
if [[ "${1:-}" == "--help" ]]; then
    python3 "$CODE_DIR/run_grid.py" prepare --help
    exit 0
fi
manifest="$(python3 "$CODE_DIR/run_grid.py" prepare "$@")"
task_count="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$manifest")"
sweep_root="$(dirname "$manifest")"
printf 'Manifest: %s\nTasks: %s\n' "$manifest" "$task_count"
sbatch --array="0-$((task_count-1))%${MAX_CONCURRENT:-1}" \
    --output="$sweep_root/logs/out_%A_%a.log" \
    --error="$sweep_root/logs/err_%A_%a.log" \
    "$CODE_DIR/experiment.sbatch" "$CODE_DIR" "$manifest"
