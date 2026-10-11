#!/bin/bash
#SBATCH --job-name=sft_compile
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%j.log
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# GPU is used only to test validation-selected removal checkpoints.
# The CSV and figure compilation itself runs on CPU.
set -eo pipefail
export SUITE_DIR="${SUITE_DIR:-$SLURM_SUBMIT_DIR/slurm/research_suite}"
source "$SUITE_DIR/config.sh"
source ~/.bashrc
conda activate "$CONDA_ENV"
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$RESULTS_ROOT/compiled_results"
python3 -u "$SUITE_DIR/../../tools/compile_research.py" \
    --root "$RESULTS_ROOT" \
    --subtraction-dir "$SUBTRACTION_DIR" \
    --evaluate-selected --device cuda "$@"
