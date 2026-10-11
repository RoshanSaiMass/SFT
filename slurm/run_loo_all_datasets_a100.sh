#!/bin/bash
#SBATCH --job-name=loo_all
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%A_%a.log
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%A_%a.log
#SBATCH --array=0-103%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# Before submitting:
# mkdir -p /export/home/achyut/Sarvesh/sflogs
# Prepare the shared dataset/backbone caches before concurrent runs.
# Submit from the repository root:
# sbatch slurm/run_loo_all_datasets_a100.sh
# Fixed 100 recovery epochs instead of early stopping:
# sbatch slurm/run_loo_all_datasets_a100.sh --epochs 100

set -eo pipefail
source ~/.bashrc
conda activate sft_env
set -u

cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

datasets=(
  pets svhn flowers102 dtd caltech101 cifar100 fgvc_aircraft
  eurosat sun397 pcam clevr dsprites-loc dsprites-ori
)

# 13 datasets x 8 PEFT configurations = 104 tasks.
# Tasks 0..7: pets; 8..15: svhn; 16..23: flowers102; and so on.
# Each task screens all 12 blocks, then trains one selected configuration.
task_id="${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}"
if (( task_id < 0 || task_id >= ${#datasets[@]} * 8 )); then
  echo "Invalid array task: ${task_id}" >&2
  exit 1
fi
dataset="${datasets[$((task_id / 8))]}"
method_id=$((task_id % 8))

case "$method_id" in
  0) name=lora
     method=(--adapter-type lora --lora-rank 16) ;;
  1) name=dora
     method=(--adapter-type dora --lora-rank 16) ;;
  2) name=paca
     method=(--adapter-type paca --lora-rank 32 --paca-tuner direct) ;;
  3) name=rpaca
     method=(--adapter-type rpaca --lora-rank 32 --paca-tuner direct) ;;
  4) name=paca_lora
     method=(--adapter-type paca --paca-tuner lora --paca-rank 32 --paca-adapter-rank 16) ;;
  5) name=paca_dora
     method=(--adapter-type paca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 16) ;;
  6) name=rpaca_lora
     method=(--adapter-type rpaca --paca-tuner lora --paca-rank 32 --paca-adapter-rank 16) ;;
  7) name=rpaca_dora
     method=(--adapter-type rpaca --paca-tuner dora --paca-rank 32 --paca-adapter-rank 16) ;;
esac

output_root="${PWD}/outputs/loo_all_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${output_root}/${dataset}/${name}"
echo "Task ${task_id}: dataset=${dataset}, PEFT=${name}"

# Early stopping by default; --epochs 100 above selects a fixed budget.
# Classifier-only warm-up adds 5 epochs before LOO screening.
# Both modes restore the best validation checkpoint before test evaluation.
python3 -u run.py loo-compact \
  --dataset "$dataset" \
  --seed 18 --device cuda \
  --num-samples 1000 --batch-size 32 \
  --filter-type symbolic --filter-rank 16 \
  --symbolic-max-terms 2 --symbolic-search-tokens 2048 \
  --loo-samples 512 --loo-batch-size 32 \
  --head-warmup-epochs 5 --head-warmup-lr 0.001 \
  --lora-rank 16 --lora-alpha 32 \
  --epochs -1 --max-epochs 200 --patience 10 \
  --data-dir "${PWD}/data" \
  --output-dir "${output_root}/${dataset}/${name}" \
  "${method[@]}" "$@"

# After the whole array finishes, compile results with:
# python3 run.py collect-all --root outputs/loo_all_JOBID --out compiled_results/loo_all_JOBID.csv
