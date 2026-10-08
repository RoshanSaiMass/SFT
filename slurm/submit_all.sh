#!/usr/bin/env bash
set -euo pipefail

# One submission covers 13 datasets × 20 configurations × the requested seeds.
CODE_DIR="${CODE_DIR:-/export/home/achyut/Sarvesh/SFT_FINAL}"
DATA_DIR="${DATA_DIR:-$CODE_DIR/data}"
CACHE_DIR="${CACHE_DIR:-$CODE_DIR/outputs/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CODE_DIR/outputs}"
PREPARE_PYTHON="${PREPARE_PYTHON:-python3}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
if [[ ! "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENT must be a positive integer" >&2; exit 2
fi
training_budget=fixed
dry_run=false
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --training-budget)
            if (( $# < 2 )); then echo "--training-budget needs fixed, early, or both" >&2; exit 2; fi
            training_budget="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash slurm/submit_all.sh [--training-budget fixed|early|both] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown option: $1 (use --help)" >&2; exit 2 ;;
    esac
done
read -r -a seed_values <<< "${SEEDS:-18}"
prepare_args=(--code-dir "$CODE_DIR" --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR"
              --output-root "$OUTPUT_ROOT" --seeds "${seed_values[@]}"
              --epochs "${EPOCHS:-100}" --max-epochs "${MAX_EPOCHS:-200}"
              --patience "${PATIENCE:-10}" --num-samples "${NUM_SAMPLES:-1000}"
              --batch-size "${BATCH_SIZE:-32}" --training-budget "$training_budget")
case "${SAVE_MISCLASSIFIED:-false}" in
    true|1|yes) prepare_args+=(--save-misclassified) ;;
    false|0|no) ;;
    *) echo "SAVE_MISCLASSIFIED must be true or false" >&2; exit 2 ;;
esac
sweep_root="$("$PREPARE_PYTHON" "$CODE_DIR/slurm/sweep.py" prepare "${prepare_args[@]}")"
manifest="$sweep_root/manifest.json"
task_count="$("$PREPARE_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["tasks"]))' "$manifest")"
command=(sbatch --array="0-$((task_count - 1))%$MAX_CONCURRENT"
         --output="$sweep_root/logs/%A_%a.out" --error="$sweep_root/logs/%A_%a.err"
         "$CODE_DIR/slurm/all_methods.sbatch" "$manifest")
printf 'Prepared %s tasks: %s\n' "$task_count" "$manifest"
printf 'Submission: '; printf '%q ' "${command[@]}"; printf '\n'
printf 'Collect later: '; printf '%q ' "$PREPARE_PYTHON" "$CODE_DIR/slurm/sweep.py" collect --manifest "$manifest"; printf '\n'
if [[ "$dry_run" == true ]]; then exit 0; fi
command -v sbatch >/dev/null || { echo "sbatch is unavailable on this host" >&2; exit 1; }
"${command[@]}"
