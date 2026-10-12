#!/bin/bash
#SBATCH --job-name=loo_all_r4
#SBATCH --output=/export/home/achyut/Sarvesh/sflogs/out_%x_%A_%a.log
#SBATCH --error=/export/home/achyut/Sarvesh/sflogs/err_%x_%A_%a.log
#SBATCH --array=0-11%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=220G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node1

# Create sflogs before submission. No research_common.sh is needed.
set -eo pipefail

EPOCHS=-1
SEED=18
CAMPAIGN="job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
DRY_RUN=0
while (($#)); do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --campaign) CAMPAIGN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ "$EPOCHS" == -1 || "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || exit 1
[[ "$SEED" =~ ^[0-9]+$ ]] || exit 1
[[ "$CAMPAIGN" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1

if [[ "$DRY_RUN" != 1 ]]; then
    source ~/.bashrc
    conda activate "${CONDA_ENV:-sft_env}"
fi
set -u

CODE_DIR="${CODE_DIR:-/export/home/achyut/Sarvesh/SFT_LOO/SFT_LOO_ALL_PEFT}"
DATA_DIR="${DATA_DIR:-/export/home/achyut/Sarvesh/SFT_BEST/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/export/home/achyut/Sarvesh/research_results}"

DATASETS=(
    pets cifar100 flowers102 dtd svhn fgvc_aircraft
    eurosat sun397 pcam clevr dsprites-loc dsprites-ori
)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
(( TASK_ID >= 0 && TASK_ID < ${#DATASETS[@]} )) || exit 1
DATASET="${DATASETS[TASK_ID]}"
OUTPUT_DIR="$RESULTS_ROOT/loo_all_rank4/$CAMPAIGN/$DATASET/seed_$SEED"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HOME="$DATA_DIR/research_cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$DATA_DIR/research_cache/torch"
export MPLCONFIGDIR="$DATA_DIR/research_cache/matplotlib"

command=(
    python3 -u "$CODE_DIR/train_loo_all_peft.py"
    --dataset "$DATASET"
    --seed "$SEED"
    --device cuda
    --num-samples 1000
    --batch-size 32
    --filter-type symbolic
    --filter-rank 16
    --symbolic-max-terms 2
    --symbolic-penalty 0.001
    --symbolic-search-tokens 2048
    --loo-samples 512
    --loo-batch-size 32
    --head-warmup-epochs 5
    --adapter-type lora
    --lora-rank 4
    --lora-alpha 8
    --epochs "$EPOCHS"
    --max-epochs 200
    --patience 10
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
)

printf '%q ' "${command[@]}"
echo
if [[ "$DRY_RUN" == 1 ]]; then
    exit 0
fi

[[ -f "$CODE_DIR/train_loo_all_peft.py" ]] || {
    echo "Missing trainer: $CODE_DIR/train_loo_all_peft.py" >&2
    exit 1
}
mkdir -p "$OUTPUT_DIR" "$DATA_DIR" "$MPLCONFIGDIR"
"${command[@]}"
