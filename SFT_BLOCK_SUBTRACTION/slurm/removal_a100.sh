#!/bin/bash
#SBATCH --job-name=sft_remove
#SBATCH --output=/export/home/achyut/Sarvesh/logs/out_%x_%A_%a.md
#SBATCH --error=/export/home/achyut/Sarvesh/logs/err_%x_%A_%a.md
#SBATCH --array=0-71%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-sft_env}"
set -u
cd "${CODE_DIR:-/export/home/achyut/Sarvesh/SFT_EXP/SFT_BLOCK_SUBTRACTION}"
: "${1:?Pass the absolute manifest.json path produced with --prepare-only --device cuda}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 -u run_removal_sweep.py --manifest "$1" --task-id "$SLURM_ARRAY_TASK_ID" --resume
