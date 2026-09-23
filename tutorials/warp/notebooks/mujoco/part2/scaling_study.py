#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 2 — Throughput scaling study
# ============================================================================
#
#  Sweeps `nworld` and reports world-steps per second so you can see where the
#  GPU stops being latency-bound and starts being genuinely parallel. Nothing
#  in this file is an exercise; it is the measurement tool used by
#  07__mujoco_warp.ipynb.
#
#  Run it:
#      python scaling_study.py
#      python scaling_study.py --worlds 1 64 1024 8192 --steps 100

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from gpu_checks import check_mjwarp_state, require_cuda
from pick_place_common import (
    apply_arm_ctrl,
    load_pick_place_model,
    reset_cubes,
    resolve_pick_place_scene,
)
from robots import add_robot_arg, get_robot

DEFAULT_WORLDS = (1, 16, 64, 256, 1024, 4096)


def measure(mjm, mjd, nworld: int, *, nconmax: int, njmax: int, steps: int) -> dict:
    """Time physics with fixed controls, excluding validation and transfers."""
    if nworld <= 0 or steps <= 0:
        raise ValueError("nworld and steps must be positive.")
    device = require_cuda()
    mjm.opt.timestep = 0.002
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

    diagnostics = check_mjwarp_state(d)  # Outside the timed region.
    return {
        "nworld": nworld,
        "timestep": float(mjm.opt.timestep),
        "solver_limit_flags": diagnostics,
        "seconds": elapsed,
        "per_step_ms": elapsed / steps * 1e3,
        "world_steps_per_s": steps * nworld / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MJWarp throughput scaling study.")
    parser.add_argument("--worlds", type=int, nargs="+", default=list(DEFAULT_WORLDS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--menagerie-path", type=Path, default=None)
    add_robot_arg(parser)
    args = parser.parse_args()
    spec = get_robot(args.robot)
    nconmax = spec.nconmax if args.nconmax is None else args.nconmax
    njmax = spec.njmax if args.njmax is None else args.njmax

    device = require_cuda()
    print(f"Warp device: {device}")

    xml_path = resolve_pick_place_scene(
        args.menagerie_path, explicit=args.menagerie_path is not None, spec=spec
    )
    mjm = load_pick_place_model(xml_path, spec)
    mjd = mujoco.MjData(mjm)
    apply_arm_ctrl(mjm, mjd, spec.home_ctrl)
    reset_cubes(mjm, mjd, spec)

    print(f"\n{spec.display_name} | physics-only, fixed controls | timestep=0.002s | warmup=10")
    print(f"{'nworld':>8} {'ms/step':>10} {'world-steps/s':>16} {'vs first batch':>14}")
    print("-" * 52)

    print(f"Throughput ratios use the first batch ({args.worlds[0]} worlds) as baseline.")
    baseline = None
    for nworld in args.worlds:
        row = measure(mjm, mjd, nworld, nconmax=nconmax, njmax=njmax, steps=args.steps)
        baseline = baseline or row["world_steps_per_s"]
        if row["solver_limit_flags"]:
            print(f"Solver iteration-limit flags for {nworld} worlds: {row['solver_limit_flags']}")
        print(
            f"{row['nworld']:>8} {row['per_step_ms']:>10.3f} "
            f"{row['world_steps_per_s']:>16,.0f} {row['world_steps_per_s'] / baseline:>13.1f}x"
        )

    print(
        "\nCompare batch latency (ms/step) with total throughput (world-steps/s).\n"
        "Scaling depends on the GPU, scene, and batch size. These fixed-control\n"
        "physics measurements exclude IK, rendering, and completed task counts."
    )


if __name__ == "__main__":
    main()
