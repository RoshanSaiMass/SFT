#!/bin/bash
#SBATCH --job-name=snip_dense
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%j.log
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

set -eo pipefail
source ~/.bashrc
conda activate sft_env

cd /export/home/achyut/Sarvesh/SFT_BEST
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python3 -u SFT_SNIP_COMPACT/train_snip_compact.py \
    --dataset pets \
    --seed 18 \
    --device cuda \
    --filter-type dense \
    --adapter-type none \
    --snip-samples 64 \
    --snip-batch-size 32 \
    --snip-search-epochs 4 \
    --snip-momentum 0.9 \
    --snip-prune-fraction 0.5 \
    --epochs -1 \
    --max-epochs 200 \
    --patience 10 \
    --data-dir /export/home/achyut/Sarvesh/SFT_BEST/data \
    --output-dir "/export/home/achyut/Sarvesh/SFT_BEST/outputs/snip_dense_baseline_${SLURM_JOB_ID}"
