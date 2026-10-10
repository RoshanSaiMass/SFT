#!/bin/bash
#SBATCH --job-name=sft_corr
#SBATCH --output=/export/home/achyut/Sarvesh/logs/out_%x_%j.md
#SBATCH --error=/export/home/achyut/Sarvesh/logs/err_%x_%j.md
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 -u train_subtraction.py --method correlation --dataset pets --adapter-type lora --lora-rank 16 --calibration-samples 64 --min-correlation 0.9 --distill-epochs 5 --seed 18 --device cuda "$@"
