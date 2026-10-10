#!/usr/bin/env bash
# Run regressions and the default fundamentals/box/preflight notebooks.
# Set NEWTON_NOTEBOOKS_INCLUDE_FINAL=1 to include the long coupled notebook.
set -euo pipefail
TUTORIAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PXR_WORK_THREAD_LIMIT=1
export NEWTON_NOTEBOOKS_EXECUTE="${NEWTON_NOTEBOOKS_EXECUTE:-1}"
export NEWTON_NOTEBOOKS_INCLUDE_FINAL="${NEWTON_NOTEBOOKS_INCLUDE_FINAL:-0}"
cd "$TUTORIAL_DIR"
if [[ $# -eq 1 && "$1" =~ ^0[1-5]$ ]]; then
    case "$1" in
        01) selector=foundations_run_headless ;;
        02) selector=both_robot_migrations_run_headless ;;
        03) selector=clean_table_runs_headless ;;
        04) selector=final_box_tasks_run_headless ;;
        05) selector=preflight_runs_in_the_selected_kernel_without_measurements ;;
    esac
    if [[ "$1" == 03 ]]; then
        export NEWTON_NOTEBOOKS_INCLUDE_FINAL=1
    fi
    python -m pytest test/test_notebooks.py test/test_benchmark_notebook.py -k "$selector"
elif [[ $# -eq 0 ]]; then
    python -m pytest test
else
    python -m pytest "$@"
fi
