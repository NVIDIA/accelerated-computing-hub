# SPDX-License-Identifier: Apache-2.0
"""Problem sizing: the memory budget per stage, machine.json, the benchmark size, the sweep sizes and the copy
bandwidth."""

import json
import subprocess

from .config import K_BINS, CANONICAL_N, SWEEP_N, MAIN_STAGES, MACHINE_PATH, RUN_INTRO_FIRST, HELD_MEMORY_HINT
from .device import device_properties, free_device_bytes, measure_copy_bandwidth_gbs
from .fields import field

# Device memory each stage holds at its peak, in bytes per voxel. A stage skips the sizes it cannot afford.
DEVICE_BYTES_PER_VOXEL = {
    "generate":     36,  # measured 32 at 512^3, rounded up
    "cupy":         48,  # the field, nine full-size temporaries and bincount's scratch
    "cupy_views":   36,  # the field, seven preallocated buffers and bincount's scratch
    "jax":          16,  # the field and XLA's temporaries
    "jax_bincount": 16,
    "cccl":          8,  # the field and CUB's scratch
    "numba":         4,  # the field and the histogram
    "cppjit_cub":   48,  # the CuPy stencil's temporaries, then CUB's scratch
    "cppjit_cuda":   4,  # the field and the histogram
}
DEVICE_FRACTION = 0.9    # the share of the device a stage may plan to hold


def _affordable(stage, sizes, device_total):
    """The sizes whose budget for `stage` fits DEVICE_FRACTION of the device."""
    return tuple(n for n in sizes if DEVICE_BYTES_PER_VOXEL[stage] * n ** 3 <= DEVICE_FRACTION * device_total)


def load_machine():
    """machine.json's contents, {} when absent."""
    return json.loads(MACHINE_PATH.read_text()) if MACHINE_PATH.exists() else {}


def sweep_sizes(stage):
    """The sizes notebook 90 recorded that `stage` can afford."""
    machine = load_machine()
    if not machine.get("sweep"):
        raise RuntimeError(f"no sweep in machine.json: {RUN_INTRO_FIRST}")
    return _affordable(stage, tuple(int(n) for n in machine["sweep"]), machine["device_total_bytes"])


def benchmark_n():
    """The shared benchmark size, the Quijote grid, as notebook 90 recorded it in machine.json."""
    machine = load_machine()
    if not machine.get("canonical"):
        raise RuntimeError(f"no machine.json: {RUN_INTRO_FIRST}")
    return int(machine["canonical"])


def copy_gbs():
    """The device copy bandwidth the intro measured, from machine.json."""
    gbs = load_machine().get("gpu_copy_gbs")
    if not gbs:
        raise RuntimeError(f"no copy bandwidth in machine.json: {RUN_INTRO_FIRST}")
    return gbs


def setup(stage, note=""):
    """(N, K) for a notebook: checks the device has room for `stage`, builds the field, prints size, device, `note`."""
    n = benchmark_n()
    need = DEVICE_BYTES_PER_VOXEL[stage] * n ** 3
    free = free_device_bytes()
    props = device_properties()
    if free < need:
        total = props["total_bytes"] / 2 ** 30
        raise RuntimeError(f"{stage} needs {need / 2 ** 30:.1f} GiB of device memory at N = {n}^3 and "
                           f"{free / 2 ** 30:.1f} GiB of {total:.0f} GiB is free: {HELD_MEMORY_HINT}")
    q = field(n)
    suffix = f"; {note}" if note else ""
    print(f"N = {n}^3, K = {K_BINS}: {q.nbytes / 2**20:.0f} MiB of int32 on {props['name']}{suffix}")
    return n, K_BINS


def save_sizing():
    """Measure the device's copy bandwidth and fix this machine's problem sizes, both recorded in machine.json.

    The sweep is SWEEP_N up to what the CuPy reference affords. The benchmark size is CANONICAL_N, the Quijote grid;
    stops when a main stage cannot afford it."""
    record = {"gpu_copy_gbs": measure_copy_bandwidth_gbs()}
    props = device_properties()
    dev = props["total_bytes"]
    print(f"  device memory              {dev / 2 ** 30:.0f} GiB")
    print(f"  device copy bandwidth      {record['gpu_copy_gbs']:.0f} GB/s (read + write)")
    sweep = _affordable("cupy", SWEEP_N, dev)
    afford = {stage: _affordable(stage, sweep, dev) for stage in MAIN_STAGES}
    for stage in MAIN_STAGES:
        if CANONICAL_N not in afford[stage]:
            raise RuntimeError(f"this device cannot hold the {CANONICAL_N}^3 Quijote cube for {stage}: "
                               f"{DEVICE_BYTES_PER_VOXEL[stage]} B/voxel at the peak")
    record["sweep"] = list(sweep)
    record["canonical"] = CANONICAL_N
    record.update({f"device_{k}": v for k, v in props.items()})
    MACHINE_PATH.write_text(json.dumps(record, indent=2))
    print(f"  benchmark size             N = {record['canonical']}^3")
    print("  sweep                      " + ", ".join(f"{n}^3" for n in sweep))
    dropped = [n for n in SWEEP_N if n not in sweep]
    if dropped:
        per_voxel = DEVICE_BYTES_PER_VOXEL["cupy"]
        need = ", ".join(f"{n ** 3 * per_voxel / 2 ** 30:.0f} GiB" for n in dropped)
        print(f"  dropped                    {', '.join(f'{n}^3' for n in dropped)}: the CuPy reference needs "
              f"{per_voxel} B/voxel on the device ({need})")
    for stage in MAIN_STAGES:
        skipped = [n for n in sweep if n not in afford[stage]]
        if skipped:
            sizes = ", ".join(f"{n}^3" for n in skipped)
            print(f"  {stage:<26} skips {sizes}: {DEVICE_BYTES_PER_VOXEL[stage]} B/voxel")


def machine_report():
    """Print the GPU, the driver, the CUDA runtime, the device memory, the L2 size, the SMs and the CuPy and JAX
    versions."""
    import cupy, jax
    props = device_properties()
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    rt = cupy.cuda.runtime.runtimeGetVersion()
    info = {"gpu": props["name"], "driver": driver, "cuda_runtime": f"{rt // 1000}.{rt % 1000 // 10}",
            "gpu_memory": f"{props['total_bytes'] / 2**30:.0f} GiB",
            "gpu_l2_cache": f"{props['l2_bytes'] / 2**20:.0f} MiB",
            "gpu_sms": f"{props['sms']} SMs x {props['threads_per_sm']} threads",
            "cupy": cupy.__version__, "jax": jax.__version__}
    for key, value in info.items():
        print(f"  {key:<14} {value}")
    return info
