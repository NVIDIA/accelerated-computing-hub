from numba import cuda  # pip install numba

import sys
import os

def get_compute_capability(device_ordinal: int = 0) -> str:
    # Check if CUDA is available
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Get list of devices
    devices = cuda.gpus.lst

    if device_ordinal < 0 or device_ordinal >= len(devices):
        raise ValueError(
            f"Invalid device ordinal {device_ordinal}; "
            f"{len(devices)} device(s) available"
        )

    # Get device and compute capability
    device = cuda.gpus[device_ordinal]
    major, minor = device.compute_capability

    # Combine into a single integer, e.g. 8 and 9 -> 89
    compute_capability = major * 10 + minor

    return (str(compute_capability) + "a") if (compute_capability == 90 or compute_capability == 100) else str(compute_capability)

def setup_local_arch(device_ordinal: int = 0):
    os.environ['PATH'] = '/usr/local/cuda-13.1/bin:' + os.environ['PATH']
    local_arch = get_compute_capability(device_ordinal)
    os.environ['LOCAL_ARCH'] = local_arch


def setup_cmake_project(device_ordinal: int = 0):
    setup_local_arch(device_ordinal)
    # Clean build directory first
    os.system("rm -rf ./build/*")
    # Configure CMake to use build/ for temporary files
    os.system("echo \"Building for ${LOCAL_ARCH}\"")
    os.system("cmake -B build -DTUTORIAL_CUDA_ARCHITECTURE=$LOCAL_ARCH")