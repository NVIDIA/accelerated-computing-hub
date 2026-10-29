#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 2 — Throughput scaling study
# ============================================================================
#
#  Sweeps `nworld` and reports world-steps per second so you can see where the
#  GPU stops being latency-bound and starts being genuinely parallel. Nothing
#  in this file is an exercise; it is the measurement tool used by
#  01_Notebook_MJWarp.ipynb.
#
#  Run it (from Article_2/part2, needs a CUDA GPU):
#      python scaling_study.py
#      python scaling_study.py --worlds 1 64 1024 2048 8192 --steps 100
#
#  The output explains every column. In short: world-steps/s is aggregate
#  physics throughput (what RL and data generation consume), compared with
#  nworld=1 on the same GPU and with one classic-MuJoCo CPU core.

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from viewer_loop import install_ctrl_c, lesson_main
from benchmark_report import (
    Row,
    measure_cpu_baseline,
    is_out_of_memory,
    mempool_peak,
    quiet_warp,
    print_header,
    print_reading_guide,
    print_row,
)
from pick_place_common import (
    apply_arm_ctrl,
    load_pick_place_model,
    reset_cubes,
    resolve_pick_place_scene,
)
from robots import add_robot_arg, get_robot
from box_task import assert_mjwarp_capacity

DEFAULT_WORLDS = (1, 16, 64, 256, 1024, 4096)


def measure(mjm, mjd, nworld: int, *, nconmax: int, njmax: int, steps: int) -> Row:
    """Time `steps` batched steps at a given world count."""
    device = wp.get_device()
    wp.synchronize()
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)

    wp.copy(d.qpos, wp.array(np.tile(mjd.qpos, (nworld, 1)), dtype=wp.float32, device=device))
    wp.copy(d.qvel, wp.array(np.tile(mjd.qvel, (nworld, 1)), dtype=wp.float32, device=device))
    wp.copy(d.ctrl, wp.array(np.tile(mjd.ctrl, (nworld, 1)), dtype=wp.float32, device=device))
    mjw.forward(m, d)

    graph = None
    if device.is_cuda:
        with wp.ScopedCapture() as capture:
            mjw.step(m, d)
        graph = capture.graph

    def advance() -> None:
        if graph is not None:
            wp.capture_launch(graph)
        else:
            mjw.step(m, d)

    for _ in range(10):
        advance()
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        advance()
    wp.synchronize()
    elapsed = time.perf_counter() - t0
    # Sticky flags retain any capacity failure during warm-up or graph replay.
    # Check outside the timed region before publishing throughput.
    assert_mjwarp_capacity(d)

    return Row(nworld=nworld, steps=steps, seconds=elapsed, mem_bytes=mempool_peak(device))


def main() -> None:
    parser = argparse.ArgumentParser(description="MJWarp throughput scaling study.")
    parser.add_argument("--worlds", type=int, nargs="+", default=list(DEFAULT_WORLDS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--menagerie-path", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Warp device, for example cuda:0 or cuda:1.")
    parser.add_argument("--fps", type=int, default=50, help="Control rate; timestep = 1 / fps / sim-substeps.")
    parser.add_argument("--sim-substeps", type=int, default=10)
    add_robot_arg(parser)
    args = parser.parse_args()
    spec = get_robot(args.robot)
    nconmax = spec.nconmax if args.nconmax is None else args.nconmax
    njmax = spec.njmax if args.njmax is None else args.njmax

    quiet_warp()
    wp.init()
    if args.device is not None:
        wp.set_device(args.device)
    device = wp.get_device()
    if not device.is_cuda:
        print("No CUDA device found — numbers below are CPU fallback, not GPU scaling.\n", file=sys.stderr)

    xml_path = resolve_pick_place_scene(
        args.menagerie_path, explicit=args.menagerie_path is not None, spec=spec
    )
    mjm = load_pick_place_model(xml_path, spec)
    # Same physics clock as the lessons (50 Hz x 10 substeps -> 2 ms).
    mjm.opt.timestep = (1.0 / args.fps) / args.sim_substeps
    mjd = mujoco.MjData(mjm)
    apply_arm_ctrl(mjm, mjd, spec.home_ctrl)
    reset_cubes(mjm, mjd, spec)
    mujoco.mj_forward(mjm, mjd)

    cpu_rate = measure_cpu_baseline(mjm, mjd)
    print_header(title="MJWarp throughput scaling study (Part 2): how many worlds per second can one GPU step?",
                 device=device, timestep=mjm.opt.timestep, steps=args.steps,
                 robot=spec.display_name, cpu_steps_per_s=cpu_rate)

    rows: list[Row] = []
    one_world = None
    for nworld in sorted(args.worlds):
        try:
            row = measure(mjm, mjd, nworld, nconmax=nconmax, njmax=njmax, steps=args.steps)
        except (RuntimeError, MemoryError) as exc:
            if not is_out_of_memory(exc):
                raise
            print(f"{nworld:>8}  out of GPU memory: this GPU cannot hold {nworld} worlds of this scene "
                  "(VRAM limit on nworld). Sweep stopped here.")
            break
        rows.append(row)
        if nworld == 1:
            one_world = row.world_steps_per_s
        print_row(row, baseline_ws=one_world, cpu_steps_per_s=cpu_rate, timestep=mjm.opt.timestep)

    print_reading_guide(rows, device=device, total_memory=getattr(device, "total_memory", None))


if __name__ == "__main__":
    install_ctrl_c()
    lesson_main(main)
