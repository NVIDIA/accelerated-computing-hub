#! /bin/bash
#
# Run tests for the stdpar tutorial.
#
# Usage:
#   ./test.bash              # run default checks
#   ./test.bash 03           # run notebook tests matching "03"
#   ./test.bash [args...]    # forward raw pytest args

TUTORIAL_ROOT=/accelerated-computing-hub/tutorials/stdpar

nvidia-smi

if [ $# -gt 0 ]; then
    if [[ "$1" == -* ]] || [[ "$1" == */* ]] || [[ "$1" == *.py ]]; then
        echo "Running: pytest $@"
        pytest "$@"
    else
        echo "Running: pytest ${TUTORIAL_ROOT}/test/test_notebooks.py -k \"$*\""
        pytest "${TUTORIAL_ROOT}/test/test_notebooks.py" -k "$*"
    fi
    exit $?
fi
