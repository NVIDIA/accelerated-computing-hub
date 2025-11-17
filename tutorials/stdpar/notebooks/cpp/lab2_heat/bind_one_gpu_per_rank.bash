#!/bin/bash
# Expose just one GPU to each MPI rank.

# --------------------------
# 1. Determine local rank
# --------------------------
if [ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]; then
    LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK        # Open MPI / HPC-X
elif [ -n "${SLURM_LOCALID:-}" ]; then
    LOCAL_RANK=$SLURM_LOCALID                     # SLURM srun
elif [ -n "${MPI_LOCALRANKID:-}" ]; then
    LOCAL_RANK=$MPI_LOCALRANKID                   # MPICH / Intel MPI
else
    echo "${0}: cannot determine local rank, defaulting to 0" >&2
    LOCAL_RANK=0
fi

# ---------------------------------------
# 2. Detect number of GPUs on this node
# ---------------------------------------
# If NGPUS is already set in the environment, respect it.
if [ -z "${NGPUS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Count lines from nvidia-smi; one line per GPU.
        NGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    else
        echo "${0}: nvidia-smi not found, assuming 1 GPU" >&2
        NGPUS=1
    fi
fi

# Safety: never let NGPUS be < 1
if [ -z "${NGPUS}" ] || [ "${NGPUS}" -lt 1 ] 2>/dev/null; then
    echo "${0}: invalid NGPUS='${NGPUS}', forcing NGPUS=1" >&2
    NGPUS=1
fi

# ---------------------------------------
# 3. Map local rank -> device index
# ---------------------------------------
DEV=$(( LOCAL_RANK % NGPUS ))

# Make only that GPU visible (as logical device 0 for this rank)
export CUDA_VISIBLE_DEVICES=${DEV}

# For NVHPC offload layers (OpenMP/OpenACC), device 0 is now the only visible one.
export OMP_DEFAULT_DEVICE=0
export ACC_DEVICE_NUM=0

# ---------------------------------------
# 4. Exec the real program
# ---------------------------------------
exec "${@}"
