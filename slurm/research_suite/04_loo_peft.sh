#!/bin/bash
#SBATCH --job-name=loo
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%A_%a.log
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%A_%a.log
#SBATCH --array=0-239%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# 12 datasets x 20 PEFT configurations; LOO-selected symbolic filter and target block; no Uni.
# Total training tasks: 240. No Caltech101.
# Recommended submission (creates directories and chunks large arrays):
# bash slurm/research_suite/submit_suite.sh --groups 'loo'
# Direct submission of this script requires creating sflogs first.
# Early stopping is default. Fixed budget: submit_suite.sh --epochs 100
set -eo pipefail
export SUITE_DIR="${SUITE_DIR:-$SLURM_SUBMIT_DIR/slurm/research_suite}"
source "$SUITE_DIR/config.sh"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    source ~/.bashrc
    conda activate "$CONDA_ENV"
fi
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$RESULTS_ROOT/loo" "$DATA_DIR" "$CACHE_DIR"
preview=()
if [[ "${DRY_RUN:-0}" == 1 ]]; then preview=(--dry-run); fi
python3 -u "$SUITE_DIR/../../tools/research_suite.py" task \
    --group 'loo' \
    --epochs "$EPOCHS" --seed "$SEED" \
    "${preview[@]}" "$@"
