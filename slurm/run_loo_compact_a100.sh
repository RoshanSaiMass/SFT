#!/bin/bash
#SBATCH --job-name=loo_compact
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%A_%a.log
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%A_%a.log
#SBATCH --array=0-7%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# Create sflogs before sbatch; prepare shared dataset/backbone caches first.
# Default early stopping: --epochs -1. Override with --epochs 100 for fixed runs.
# Override dataset with: sbatch slurm/run_loo_compact_a100.sh --dataset cifar100
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-sft_env}"
set -u
cd "${CODE_DIR:-/export/home/achyut/Sarvesh/SFT_BEST}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) method=(--adapter-type lora) ;;
  1) method=(--adapter-type dora) ;;
  2) method=(--adapter-type paca --lora-rank 32 --paca-tuner direct) ;;
  3) method=(--adapter-type rpaca --lora-rank 32 --paca-tuner direct) ;;
  4) method=(--adapter-type paca --paca-tuner lora --paca-rank 32 --paca-adapter-rank 16) ;;
  5) method=(--adapter-type paca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 16) ;;
  6) method=(--adapter-type rpaca --paca-tuner lora --paca-rank 32 --paca-adapter-rank 16) ;;
  7) method=(--adapter-type rpaca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 16) ;;
  *) echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

python3 -u run.py loo-compact \
  --dataset pets --seed 18 --device cuda \
  --filter-type symbolic --filter-rank 16 \
  --loo-samples 256 --loo-batch-size 32 \
  --head-warmup-epochs 5 --head-warmup-lr 0.001 \
  --lora-rank 16 --epochs -1 --max-epochs 200 --patience 10 \
  --data-dir "${PWD}/data" \
  --output-dir "${PWD}/outputs/loo_compact_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}" \
  "${method[@]}" "$@"
