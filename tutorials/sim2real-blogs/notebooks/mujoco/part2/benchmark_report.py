# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 2 — Self-explanatory throughput output (not an exercise)
# ============================================================================
#
#  Used by `so101_mjwarp*.py --benchmark` and `scaling_study.py` so that the
#  printed numbers say what they measure, in which units, against which
#  baseline, and how they are expected to change on other hardware.

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass

import mujoco
import numpy as np

RULE = "-" * 96


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_description(device) -> str:
    if not getattr(device, "is_cuda", False):
        return f"{device} (no CUDA device: Warp is running on the CPU, numbers are NOT GPU numbers)"
    gib = getattr(device, "total_memory", 0) / 2**30
    sms = getattr(device, "sm_count", "?")
    return f"{device.name} ({device}), {sms} SMs, {gib:.0f} GiB VRAM, compute capability {device.arch // 10}.{device.arch % 10}"


def measure_cpu_baseline(mjm, mjd, *, seconds: float = 1.0, min_steps: int = 200) -> float:
    """Single-thread classic MuJoCo `mj_step` rate on the same model and state.

    This is the reference that "speed-up vs CPU" divides by: one CPU core
    advancing one world. A multi-process CPU farm scales roughly with cores;
    Part 2's `migration_benchmark.py` measures that properly (up to 32 workers).
    """
    data = mujoco.MjData(mjm)
    data.qpos[:], data.qvel[:], data.ctrl[:] = mjd.qpos, mjd.qvel, mjd.ctrl
    mujoco.mj_forward(mjm, data)
    for _ in range(20):  # warm caches
        mujoco.mj_step(mjm, data)
    steps, t0 = 0, time.perf_counter()
    while steps < min_steps or time.perf_counter() - t0 < seconds:
        for _ in range(50):
            mujoco.mj_step(mjm, data)
        steps += 50
    return steps / (time.perf_counter() - t0)


@dataclass
class Row:
    nworld: int
    steps: int
    seconds: float
    mem_bytes: int | None = None

    @property
    def ms_per_step(self) -> float:
        return self.seconds / self.steps * 1e3

    @property
    def world_steps_per_s(self) -> float:
        return self.steps * self.nworld / self.seconds


def print_header(*, title: str, device, timestep: float, steps: int, robot: str, cpu_steps_per_s: float) -> None:
    print(RULE)
    print(title)
    print(RULE)
    print(f"Robot/scene : {robot}, arm holding its start pose over the cubes (controls fixed, no IK, no viewer)")
    print(f"GPU         : {gpu_description(device)}")
    print(f"CPU         : {cpu_name()} ({os.cpu_count()} logical cores)")
    print(f"Physics     : timestep {timestep * 1e3:.1f} ms; one 'step' = one mjw.step call that advances EVERY world by one timestep")
    print(f"Timing      : {steps} CUDA-graph replays after 10 warm-up steps, wall clock between two wp.synchronize() calls")
    print(f"CPU baseline: classic MuJoCo mj_step, ONE world on ONE core = {cpu_steps_per_s:,.0f} steps/s "
          f"(real-time factor {cpu_steps_per_s * timestep:,.1f}x)")
    print(RULE)
    print("Columns:")
    print("  nworld       independent copies of the scene simulated together in one batch")
    print("  ms/step      wall-clock milliseconds for one batched step (latency of the whole batch)")
    print("  world-steps/s  nworld x steps / seconds: aggregate throughput, i.e. physics samples per second")
    print("  vs 1 world   throughput relative to nworld=1 on the same GPU")
    print("  vs CPU core  throughput relative to the single-core CPU baseline above")
    print("  sim-s/s      simulated seconds produced per wall-clock second over all worlds (real-time factor)")
    print("  peak VRAM    Warp memory high-water mark for this batch, temporaries included (+ ~0.5 GiB CUDA context)")
    print(RULE)
    print(f"{'nworld':>8} {'ms/step':>9} {'world-steps/s':>15} {'vs 1 world':>11} {'vs CPU core':>12} {'sim-s/s':>11} {'peak VRAM':>10}")


def print_row(row: Row, *, baseline_ws: float | None, cpu_steps_per_s: float, timestep: float) -> None:
    ws = row.world_steps_per_s
    vs_one = f"{ws / baseline_ws:>10.1f}x" if baseline_ws else f"{'—':>11}"
    mem = f"{row.mem_bytes / 2**20:>7.0f} MiB" if row.mem_bytes is not None else f"{'n/a':>10}"
    print(f"{row.nworld:>8} {row.ms_per_step:>9.3f} {ws:>15,.0f} {vs_one} {ws / cpu_steps_per_s:>11.1f}x "
          f"{ws * timestep:>11,.0f} {mem}")


def print_reading_guide(rows: list[Row], *, device, total_memory: int | None = None) -> None:
    print(RULE)
    print("How to read this:")
    print("  * Throughput (world-steps/s) is what matters for reinforcement learning and synthetic data:")
    print("    it is how many physics samples per second the simulator can feed a learner.")
    print("  * ms/step grows much more slowly than nworld while the GPU still has idle SMs, so throughput")
    print("    rises almost linearly. Once the SMs and memory bandwidth are saturated, ms/step grows in")
    print("    proportion to nworld and throughput plateaus: that knee is this GPU's useful batch size.")
    print("  * One world on a GPU is usually SLOWER than one CPU core (kernel launch latency, float32 batch")
    print("    machinery); MJWarp pays off only with many worlds.")
    print("  * 'vs CPU core' compares with ONE core. A CPU farm runs one world per core, so divide by the cores")
    print("    you would use (e.g. 32) for a fair farm comparison; BENCHMARK.md measures exactly that on the")
    print("    full task, where the GPU overtakes 32 CPU workers from about 1,024 worlds.")
    if rows and all(r.mem_bytes for r in rows) and total_memory:
        big = max(rows, key=lambda r: r.nworld)
        per_world = big.mem_bytes / big.nworld
        print(f"  * Memory: about {per_world / 2**20:,.2f} MiB per world at the peak (contact/constraint buffers sized by")
        print("    nconmax/njmax, plus collision temporaries). VRAM is the hard cap on nworld. Rough, conservative")
        fits = ", ".join(f"{label} ~{0.6 * gib * 2**30 / per_world:,.0f}"
                         for label, gib in (("12 GB card", 12), ("32 GB card", 32),
                                            (f"this {total_memory / 2**30:.0f} GiB GPU", total_memory / 2**30)))
        print(f"    largest batch: {fits} worlds. 'out of memory' means: lower nworld or nconmax/njmax.")
    print("  * Other hardware: plateau throughput scales roughly with SM count x clock and memory bandwidth;")
    print("    the largest nworld is capped by VRAM; the CPU baseline depends on single-core speed.")
    print("    A smaller GPU (e.g. RTX 3060, 28 SMs, 12 GB) reaches its plateau at fewer worlds and lower")
    print("    throughput; a larger one (e.g. RTX 5090, 170 SMs, 32 GB) keeps scaling to larger batches.")
    print("  * This is a physics-step microbenchmark with fixed controls. The full two-cube task, with")
    print("    validation and matched CPU worker processes, is measured by migration_benchmark.py (see BENCHMARK.md).")
    print(RULE)


def quiet_warp() -> None:
    """Hide Warp's per-module load lines so the table stays readable."""
    import warp as wp
    if hasattr(wp, "LOG_WARNING") and hasattr(wp.config, "log_level"):
        wp.config.log_level = wp.LOG_WARNING
    else:
        wp.config.quiet = True


def mempool_peak(device) -> int | None:
    """Warp memory-pool high-water mark (bytes) on ``device`` since start-up.

    It includes the temporary buffers a step allocates (collision, constraint
    solver), which the persistent model/data arrays alone would under-count.
    Sweeps run in ascending nworld, so the mark after a row is that row's peak.
    """
    import warp as wp
    try:
        if device.is_cuda and wp.is_mempool_enabled(device):
            return int(wp.get_mempool_used_mem_high(device))
    except Exception:  # noqa: BLE001 - memory reporting is optional
        return None
    return None


def is_out_of_memory(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "failed to allocate" in text or "alloc" in text and "cuda" in text


def tile_state(mjd, nworld: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.tile(mjd.qpos, (nworld, 1)), np.tile(mjd.qvel, (nworld, 1)), np.tile(mjd.ctrl, (nworld, 1)))
