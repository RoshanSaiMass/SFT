#!/bin/bash
#SBATCH --job-name=loo_bottleneck
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
set -eo pipefail
REPO_ROOT="${BOTTLENECK_REPO:-/export/home/achyut/Sarvesh/SFT_BOTTLENECK}"
DATA_DIR="${DATA_DIR:-/export/home/achyut/Sarvesh/SFT_BEST/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/export/home/achyut/Sarvesh/research_results}"
FILTER_RANK=64
PEFT_RANK=8
ADAPTER=lora
PACA_RANK=32
PACA_TUNER=direct
EPOCHS=-1
SEED=18
CAMPAIGN="bottleneck_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
DRY_RUN=0
while (($#)); do
    case "$1" in
        --filter-rank) FILTER_RANK="$2"; shift 2 ;;
        --peft-rank) PEFT_RANK="$2"; shift 2 ;;
        --adapter-type) ADAPTER="$2"; shift 2 ;;
        --paca-rank) PACA_RANK="$2"; shift 2 ;;
        --paca-tuner) PACA_TUNER="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --campaign) CAMPAIGN="$2"; shift 2 ;;
        --repo-root) REPO_ROOT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ "$CAMPAIGN" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1
[[ "$FILTER_RANK" =~ ^[1-9][0-9]*$ && "$PEFT_RANK" =~ ^[1-9][0-9]*$ && "$PACA_RANK" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || exit 1
(( FILTER_RANK <= 768 )) || exit 1
[[ "$ADAPTER" =~ ^(none|lora|dora|paca|rpaca)$ && "$PACA_TUNER" =~ ^(direct|lora|dora)$ ]] || exit 1
[[ "$EPOCHS" == -1 || "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || exit 1
DATASETS=(pets cifar100 flowers102 dtd svhn fgvc_aircraft eurosat sun397 pcam clevr dsprites-loc dsprites-ori)
ID="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$ID" =~ ^[0-9]+$ ]] || exit 1
(( ID < 12 )) || exit 1
DATASET="${DATASETS[ID]}"
CODE_DIR="$REPO_ROOT/SFT_LOO_BOTTLENECK"
[[ -f "$CODE_DIR/train_bottleneck.py" ]] || { echo "Incorrect repository root: $REPO_ROOT" >&2; exit 1; }
OUTPUT_DIR="$RESULTS_ROOT/loo_bottleneck/$CAMPAIGN/$DATASET/${ADAPTER}_${PACA_TUNER}_f${FILTER_RANK}_r${PEFT_RANK}_c${PACA_RANK}_s${SEED}"
CACHE_DIR="$DATA_DIR/research_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
COMMAND=(python3 -u "$CODE_DIR/train_bottleneck.py"
    --dataset "$DATASET" --seed "$SEED" --device cuda
    --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"
    --filter-rank "$FILTER_RANK" --adapter-type "$ADAPTER"
    --lora-rank "$PEFT_RANK" --lora-alpha "$((2 * PEFT_RANK))"
    --paca-rank "$PACA_RANK" --paca-tuner "$PACA_TUNER" --paca-adapter-rank "$PEFT_RANK"
    --bottleneck-init-steps 200 --bottleneck-init-lr 0.001 --init-tokens 2048
    --loo-samples 512 --loo-batch-size 32 --head-warmup-epochs 5
    --num-samples 1000 --batch-size 32
    --checkpoint-format trainable --epochs "$EPOCHS" --max-epochs 200 --patience 10)
printf '%q ' "${COMMAND[@]}"
printf '\n'
[[ "$DRY_RUN" != 1 ]] || exit 0
source ~/.bashrc
conda activate sft_env
mkdir -p "$OUTPUT_DIR" "$DATA_DIR"
python3 "$REPO_ROOT/tools/prepare_num_function.py" --code-dir "$CODE_DIR" \
    --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" --dataset "$DATASET" --seed "$SEED"
"${COMMAND[@]}"
