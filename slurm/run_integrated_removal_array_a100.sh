#!/bin/bash
#SBATCH --job-name=sft_remove
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%A_%a.md
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%A_%a.md
#SBATCH --array=0-71%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# Pass manifest.json prepared with run.py removal-sweep --prepare-only --device cuda.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-sft_env}"
set -u
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
: "${1:?Pass the absolute manifest.json path}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 -u run.py removal-sweep --manifest "$1" --task-id "$SLURM_ARRAY_TASK_ID" --resume
