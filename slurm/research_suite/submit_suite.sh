#!/bin/bash
# Run on the login node from this repository's root:
# bash slurm/research_suite/submit_suite.sh
# Only one experiment:
# bash slurm/research_suite/submit_suite.sh --groups snip_only
# Fixed 100 epochs:
# bash slurm/research_suite/submit_suite.sh --epochs 100
# Preview submissions without submitting:
# bash slurm/research_suite/submit_suite.sh --dry-run
set -eo pipefail
export SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SUITE_DIR/config.sh"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    source ~/.bashrc
    conda activate "$CONDA_ENV"
fi
set -u
python3 "$SUITE_DIR/../../tools/research_suite.py" submit --epochs "$EPOCHS" --seed "$SEED" "$@"
