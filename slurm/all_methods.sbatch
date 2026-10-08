#!/usr/bin/env bash
#SBATCH --job-name=sft-all-methods
#SBATCH --partition=gpu-a100
#SBATCH --nodelist=node1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=11:59:59

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit this script through slurm/submit_all.sh}"
manifest="${1:?The submission helper must pass manifest.json}"
bootstrap_python="$(command -v python3 || command -v python)"
code_dir="$("$bootstrap_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_dir"])' "$manifest")"
record_setup_failure() {
    local rc=$?
    if (( rc != 0 )); then
        "$bootstrap_python" "$code_dir/slurm/sweep.py" record-failure --manifest "$manifest" \
            --task-id "$SLURM_ARRAY_TASK_ID" --returncode "$rc" \
            --error "Job environment setup failed; see the SLURM task error log" || true
    fi
}
trap record_setup_failure EXIT
CONDA_ENV="${CONDA_ENV:-paca_env}"
if [[ -n "${CONDA_SH:-}" ]]; then
    source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
    source "$conda_base/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda was not found; export CONDA_SH=/absolute/path/etc/profile.d/conda.sh before submitting" >&2
    exit 1
fi
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
exec python "$code_dir/slurm/sweep.py" task --manifest "$manifest" --task-id "$SLURM_ARRAY_TASK_ID"
