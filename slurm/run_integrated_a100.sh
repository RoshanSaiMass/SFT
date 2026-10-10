#!/bin/bash
#SBATCH --job-name=sft_integrated
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%A_%a.md
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%A_%a.md
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# Submit from the directory containing run.py, or export CODE_DIR.
# Create /export/home/achyut/Sarvesh/sflogs before submission.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-sft_env}"
set -u
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 -u run.py "$@"
