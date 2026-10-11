#!/bin/bash
# Edit these paths if a trainer lives somewhere else on your cluster.
export CLUSTER_BASE="${CLUSTER_BASE:-/export/home/achyut/Sarvesh}"
export SNIP_DIR="${SNIP_DIR:-$CLUSTER_BASE/SFT_BEST/SFT_SNIP_COMPACT}"
export ORIGINAL_DIR="${ORIGINAL_DIR:-$CLUSTER_BASE/SFT_FINAL}"
export SYMBOLIC_DIR="${SYMBOLIC_DIR:-$CLUSTER_BASE/SFT_EXP/SFT_EXPERIEMTN}"
export LOO_DIR="${LOO_DIR:-$CLUSTER_BASE/SFT_LOO/SFT_LOO_COMPACT}"
export SUBTRACTION_DIR="${SUBTRACTION_DIR:-$CLUSTER_BASE/SFT_COEFF_RED/SFT_BLOCK_SUBTRACTION}"
export RESULTS_ROOT="${RESULTS_ROOT:-$CLUSTER_BASE/research_results}"
export DATA_DIR="${DATA_DIR:-$CLUSTER_BASE/SFT_BEST/data}"
export CACHE_DIR="${CACHE_DIR:-$DATA_DIR/research_cache}"
export CONDA_ENV="${CONDA_ENV:-sft_env}"
export EPOCHS="${EPOCHS:--1}"
export SEED="${SEED:-18}"
