#! /bin/bash
#
# Run tests for the cuda-cpp tutorial.
#
# Usage:
#   ./test.bash              # run default checks
#   ./test.bash 03           # run notebook tests matching "03"
#   ./test.bash [args...]    # forward raw pytest args

TUTORIAL_ROOT=/accelerated-computing-hub/tutorials/cuda-cpp

START_TIME=$(date +%s.%N)

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || exit 1
else
    NVIDIA_GPU_DEVICE=$(find /dev -maxdepth 1 -type c \
        -name 'nvidia[0-9]*' -print -quit 2>/dev/null)
    if [ -n "${NVIDIA_GPU_DEVICE}" ]; then
        echo "NVIDIA GPU device ${NVIDIA_GPU_DEVICE} is available; nvidia-smi is not installed"
    else
        echo "Error: no NVIDIA GPU is available" >&2
        exit 1
    fi
fi

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

END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {print $END_TIME - $START_TIME}")

echo ""
awk -v elapsed="$ELAPSED" 'BEGIN {
    hours = int(elapsed / 3600)
    minutes = int((elapsed % 3600) / 60)
    seconds = elapsed % 60
    printf "Elapsed time: %dh %dm %.3fs\n", hours, minutes, seconds
}'

exit 0
