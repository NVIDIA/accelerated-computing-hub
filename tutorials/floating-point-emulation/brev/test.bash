#! /bin/bash

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found; continuing without GPU inventory"
fi
