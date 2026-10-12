#!/bin/bash
#SBATCH --job-name=sub_correlation
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%A_%a.log
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%A_%a.log
#SBATCH --array=0-263%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

set -eo pipefail
found=0
for candidate in "${SCHEDULER_DIR:-}" "${REPO_ROOT:-}/scheduler"     "${SLURM_SUBMIT_DIR:-$PWD}/scheduler" "${SLURM_SUBMIT_DIR:-$PWD}"     /export/home/achyut/Sarvesh/SFT_FINAL_RESEARCH/scheduler; do
    if [[ -f "$candidate/research_common.sh" && -f "$candidate/../tools/research_scheduler.py" ]]; then
        export SCHEDULER_DIR="$candidate"
        found=1
        break
    fi
done
if [[ "$found" != 1 ]]; then
    echo "Cannot locate scheduler. Submit from the repository root or export SCHEDULER_DIR=/absolute/repo/scheduler." >&2
    exit 1
fi
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then export DRY_RUN=1; fi
done
source "$SCHEDULER_DIR/research_common.sh"
python3 -u "$REPO_ROOT/tools/research_scheduler.py" task --group "sub_correlation" "$@"
