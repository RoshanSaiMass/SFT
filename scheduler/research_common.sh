#!/bin/bash
# Sourced by launchers; do not submit this file with sbatch.
export REPO_ROOT="$(cd "$SCHEDULER_DIR/.." && pwd)"
export CLUSTER_BASE="${CLUSTER_BASE:-/export/home/achyut/Sarvesh}"
export RESULTS_ROOT="${RESULTS_ROOT:-$CLUSTER_BASE/research_results}"
export DATA_DIR="${DATA_DIR:-$CLUSTER_BASE/SFT_BEST/data}"
export CACHE_DIR="${CACHE_DIR:-$DATA_DIR/research_cache}"
export CONDA_ENV="${CONDA_ENV:-sft_env}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    source ~/.bashrc
    conda activate "$CONDA_ENV"
fi
