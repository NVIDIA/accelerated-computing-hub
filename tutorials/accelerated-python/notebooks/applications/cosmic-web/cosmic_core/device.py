# SPDX-License-Identifier: Apache-2.0
"""Device properties, free memory, the CuPy memory pool and the device copy bandwidth."""

import numpy as np

from .config import WARMUP


def release_pool():
    """Return the CuPy pool's unused blocks to the device, so this process holds only its live arrays."""
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()


def device_properties():
    """Name, total memory, L2 size, SM count and threads per SM of the current device."""
    import cupy
    p = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    return {"name": p["name"].decode(), "total_bytes": int(p["totalGlobalMem"]), "l2_bytes": int(p["l2CacheSize"]),
            "sms": int(p["multiProcessorCount"]), "threads_per_sm": int(p["maxThreadsPerMultiProcessor"])}


def free_device_bytes():
    """Device memory free at this moment, in bytes."""
    import cupy
    return int(cupy.cuda.runtime.memGetInfo()[0])


def measure_copy_bandwidth_gbs():
    """Device-to-device copy bandwidth of a 1 GiB copy in GB/s, read and write counted."""
    import cupy as cp
    nbytes, repeats = 1 << 30, 10
    src = cp.ones(nbytes, dtype=cp.uint8)
    dst = cp.empty_like(src)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for i in range(WARMUP + repeats):
        start.record()
        cp.copyto(dst, src)
        stop.record()
        stop.synchronize()
        if i >= WARMUP:
            times.append(cp.cuda.get_elapsed_time(start, stop) * 1e-3)
    del src, dst
    release_pool()
    return float(2.0 * nbytes / np.median(times) / 1e9)
