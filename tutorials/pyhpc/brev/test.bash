#! /bin/bash
#
# Run tests for the pyhpc tutorial.
#
# When called with no arguments, runs both test suites (packages, notebooks).
# When called with arguments:
#   - Bare words (e.g. "06") are treated as a pytest -k filter for notebook tests.
#   - Paths or flags (e.g. "test/test_packages.py", "-k foo") are forwarded to
#     pytest directly.
#
# Usage:
#   ./test.bash                          # run all suites
#   ./test.bash 06                       # run notebook tests matching "06"
#   ./test.bash "06 or 07"              # run notebook tests matching "06 or 07"
#   ./test.bash test/test_packages.py    # run only package tests
#   ./test.bash -k "cupy"               # forward raw pytest flags

START_TIME=$(date +%s.%N)

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found; continuing without GPU inventory"
fi

TUTORIAL_ROOT=/accelerated-computing-hub/tutorials/pyhpc

if [ $# -gt 0 ]; then
    if [[ "$1" == -* ]] || [[ "$1" == */* ]] || [[ "$1" == *.py ]]; then
        echo "Running: pytest $@"
        pytest "$@"
    else
        echo "Running: pytest ${TUTORIAL_ROOT}/test/test_notebooks.py -k \"$*\""
        pytest "${TUTORIAL_ROOT}/test/test_notebooks.py" -k "$*"
    fi
    EXIT_CODE=$?
else
    # Run package tests.
    echo "Running package tests..."
    pytest "${TUTORIAL_ROOT}/test/test_packages.py"
    EXIT_CODE_PACKAGES=$?

    # Test notebooks (solutions where they exist, exercises otherwise).
    echo ""
    echo "Running notebook tests..."
    pytest "${TUTORIAL_ROOT}/test/test_notebooks.py"
    EXIT_CODE_NOTEBOOKS=$?

    # Overall exit code is non-zero if any test suite failed.
    EXIT_CODE=$((EXIT_CODE_PACKAGES || EXIT_CODE_NOTEBOOKS))
fi

END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {print $END_TIME - $START_TIME}")

echo ""
awk -v elapsed="$ELAPSED" 'BEGIN {
    hours = int(elapsed / 3600)
    minutes = int((elapsed % 3600) / 60)
    seconds = elapsed % 60
    printf "Elapsed time: %dh %dm %.3fs\n", hours, minutes, seconds
}'

exit $EXIT_CODE
