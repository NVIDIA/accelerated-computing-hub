#!/usr/bin/env bash
# Run regression checks plus the two short reference notebooks.
# Set NEWTON_NOTEBOOKS_INCLUDE_FINAL=1 for all four (long CPU gripper tasks).
set -euo pipefail
TUTORIAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PXR_WORK_THREAD_LIMIT=1
export NEWTON_NOTEBOOKS_EXECUTE="${NEWTON_NOTEBOOKS_EXECUTE:-1}"
export NEWTON_NOTEBOOKS_INCLUDE_FINAL="${NEWTON_NOTEBOOKS_INCLUDE_FINAL:-0}"
cd "$TUTORIAL_DIR"
if [[ $# -eq 1 && "$1" =~ ^0[1-4]$ ]]; then
    case "$1" in
        01) selector=foundations_run_headless ;;
        02) selector=both_robot_migrations_run_headless ;;
        03) selector=clean_table_runs_headless ;;
        04) selector=final_includes_both_full_clean_table_tasks ;;
    esac
    if [[ "$1" == 03 || "$1" == 04 ]]; then
        export NEWTON_NOTEBOOKS_INCLUDE_FINAL=1
    fi
    python -m pytest test/test_notebooks.py -k "$selector"
elif [[ $# -eq 0 ]]; then
    python -m pytest test
else
    python -m pytest "$@"
fi
