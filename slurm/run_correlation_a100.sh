#!/bin/bash
#SBATCH --job-name=sft_corr
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%j.log
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# Create the log directory BEFORE submitting:
# mkdir -p /export/home/achyut/Sarvesh/sflogs
# Early stopping: sbatch slurm/run_correlation_a100.sh
# Fixed 100 epochs: sbatch slurm/run_correlation_a100.sh --epochs 100

set -eo pipefail
source ~/.bashrc
conda activate sft_env
set -u

# Submit from the repository root, or export CODE_DIR.
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Find the highest-Pearson adjacent pair; no 0.9 cutoff.
# Remove the later block and apply LoRA to the earlier block.
# Five feature-distillation epochs precede classification training.
# Restore the best validation checkpoint, then evaluate test.
python3 -u run.py correlation \
  --dataset pets \
  --seed 18 --device cuda \
  --pair-index -1 \
  --min-correlation -1 \
  --calibration-samples 64 \
  --adapter-scope retained \
  --adapter-type lora \
  --lora-rank 16 --lora-alpha 32 \
  --train-layernorms all \
  --distill-epochs 5 \
  --distill-lr 0.0001 \
  --distill-batch-size 8 \
  --epochs -1 --max-epochs 200 --patience 10 \
  --data-dir "${PWD}/data" \
  --output-dir "${PWD}/outputs/correlation_${SLURM_JOB_ID}" \
  "$@"
