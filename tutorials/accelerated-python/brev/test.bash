#! /bin/bash

START_TIME=$(date +%s.%N)

nvidia-smi

# Run tests.
echo "Running tests..."
pytest /accelerated-computing-hub/tutorials/accelerated-python/test/ \
  --ignore=/accelerated-computing-hub/tutorials/accelerated-python/test/test_rapids.py
EXIT_CODE_TESTS=$?

# Run RAPIDS tests separately because they require a different virtual environment.
echo ""
echo "Running RAPIDS tests in virtual environment..."
/opt/venvs/rapids/bin/pytest /accelerated-computing-hub/tutorials/accelerated-python/test/test_rapids.py
EXIT_CODE_RAPIDS=$?

# Overall exit code is non-zero if any test suite failed.
EXIT_CODE=$((EXIT_CODE_TESTS || EXIT_CODE_RAPIDS))

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
