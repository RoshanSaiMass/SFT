#!/bin/bash
#SBATCH --job-name=sft_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1
set -eo pipefail
if [[ -n "${CONDA_SH:-}" ]]; then
    source "$CONDA_SH"
else
    source ~/.bashrc
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-sft_env}"
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$1"
python3 run_grid.py run --manifest "$2" --task-id "$SLURM_ARRAY_TASK_ID"
