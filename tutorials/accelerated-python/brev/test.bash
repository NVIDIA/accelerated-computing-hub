#! /bin/bash

START_TIME=$(date +%s.%N)

nvidia-smi

# Run regular package tests.
echo "Running regular package tests..."
pytest /accelerated-computing-hub/tutorials/accelerated-python/test/test_packages.py
EXIT_CODE_PACKAGES=$?

# Run RAPIDS tests.
echo ""
echo "Running RAPIDS package tests in virtual environment..."
/opt/venvs/rapids/bin/pytest /accelerated-computing-hub/tutorials/accelerated-python/test/test_rapids.py
EXIT_CODE_RAPIDS=$?

# Test solution notebooks.
echo ""
echo "Running solution notebook tests..."
pytest /accelerated-computing-hub/tutorials/accelerated-python/test/test_notebooks.py
EXIT_CODE_NOTEBOOKS=$?

# Overall exit code is non-zero if any test suite failed.
EXIT_CODE=$((EXIT_CODE_PACKAGES || EXIT_CODE_RAPIDS || EXIT_CODE_NOTEBOOKS))

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
