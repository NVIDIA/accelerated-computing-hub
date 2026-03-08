#! /bin/bash
#
# Run tests for the accelerated-python tutorial.
#
# When called with no arguments, runs all three test suites (packages, RAPIDS,
# notebooks).  When called with arguments:
#   - Bare words (e.g. "03") are treated as a pytest -k filter for notebook tests.
#   - Paths or flags (e.g. "test/test_packages.py", "-k foo") are forwarded to
#     pytest directly.
#
# Usage:
#   ./test.bash                          # run all suites
#   ./test.bash 03                       # run notebook tests matching "03"
#   ./test.bash "05 or 06"              # run notebook tests matching "05 or 06"
#   ./test.bash test/test_packages.py    # run only package tests
#   ./test.bash -k "cupy"               # forward raw pytest flags

START_TIME=$(date +%s.%N)

nvidia-smi

TUTORIAL_ROOT=/accelerated-computing-hub/tutorials/accelerated-python

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
    # Run regular package tests.
    echo "Running regular package tests..."
    pytest "${TUTORIAL_ROOT}/test/test_packages.py"
    EXIT_CODE_PACKAGES=$?

    # Run RAPIDS tests.
    echo ""
    echo "Running RAPIDS package tests in virtual environment..."
    /opt/venvs/rapids/bin/pytest "${TUTORIAL_ROOT}/test/test_rapids.py"
    EXIT_CODE_RAPIDS=$?

    # Test solution notebooks.
    echo ""
    echo "Running solution notebook tests..."
    pytest "${TUTORIAL_ROOT}/test/test_notebooks.py"
    EXIT_CODE_NOTEBOOKS=$?

    # Overall exit code is non-zero if any test suite failed.
    EXIT_CODE=$((EXIT_CODE_PACKAGES || EXIT_CODE_RAPIDS || EXIT_CODE_NOTEBOOKS))
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
