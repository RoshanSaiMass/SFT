#!/bin/bash
#SBATCH --job-name=sft_remove
#SBATCH --output=/home/achyutm01/SFT/logs/out_%x_%A_%a.md
#SBATCH --error=/home/achyutm01/SFT/logs/err_%x_%A_%a.md
#SBATCH --array=0-71%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=64G
#SBATCH --partition=mig90g
#SBATCH --gres=gpu:3g.90gb:1

set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-sft_env}"
set -u
cd "${CODE_DIR:-/home/achyutm01/SFT/SFT_BLOCK_SUBTRACTION}"
: "${1:?Pass the absolute manifest.json path produced with --prepare-only --device cuda}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 -u run_removal_sweep.py --manifest "$1" --task-id "$SLURM_ARRAY_TASK_ID" --resume
