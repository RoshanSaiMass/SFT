#!/bin/bash
set -eo pipefail
export SCHEDULER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arg in "$@"; do
    if [[ "$arg" == --dry-run ]]; then export DRY_RUN=1; fi
done
source "$SCHEDULER_DIR/research_common.sh"
python3 "$REPO_ROOT/tools/research_scheduler.py" submit "$@"
