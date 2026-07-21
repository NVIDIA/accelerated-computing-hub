#! /bin/bash

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
