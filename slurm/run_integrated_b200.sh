#!/bin/bash
#SBATCH --job-name=sft_integrated
#SBATCH --output=/home/achyutm01/SFT/sflogs/out_%x_%A_%a.md
#SBATCH --error=/home/achyutm01/SFT/sflogs/err_%x_%A_%a.md
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=11:59:59
#SBATCH --mem=64G
#SBATCH --partition=mig90g
#SBATCH --gres=gpu:3g.90gb:1

# Submit from the directory containing run.py, or export CODE_DIR.
# Create /home/achyutm01/SFT/sflogs before submission.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-sft_env}"
set -u
cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 -u run.py "$@"
