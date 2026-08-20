#! /bin/bash
#
# Run tests for the cuda-tile tutorial.

set -euo pipefail

START_TIME=$(date +%s.%N)

nvidia-smi

TUTORIAL_ROOT=/accelerated-computing-hub/tutorials/cuda-tile

if [ $# -gt 0 ]; then
    echo "Running: $*"
    "$@"
else
    NOTEBOOK="${TUTORIAL_ROOT}/notebooks/01__cutile_python_intro__vector_add.ipynb"
    OUTPUT="/tmp/01__cutile_python_intro__vector_add.executed.ipynb"

    echo "Running: jupyter nbconvert --execute ${NOTEBOOK}"
    jupyter nbconvert \
        --to notebook \
        --execute "${NOTEBOOK}" \
        --output "${OUTPUT}" \
        --ExecutePreprocessor.timeout=900
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
