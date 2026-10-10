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
#  01_Notebook_MJWarp.ipynb.
#
#  Run it:
#      python scaling_study.py
#      python scaling_study.py --worlds 1 64 1024 8192 --steps 100

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from pick_place_common import (
    apply_arm_ctrl,
    load_pick_place_model,
    reset_cubes,
    resolve_pick_place_scene,
)
from robots import add_robot_arg, get_robot

DEFAULT_WORLDS = (1, 16, 64, 256, 1024, 4096)


def measure(mjm, mjd, nworld: int, *, nconmax: int, njmax: int, steps: int) -> dict:
    """Time `steps` batched steps at a given world count."""
    device = wp.get_device()
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

    return {
        "nworld": nworld,
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

    wp.init()
    device = wp.get_device()
    print(f"Warp device: {device}")
    if not device.is_cuda:
        print("No CUDA device found — numbers below are CPU fallback, not GPU scaling.\n", file=sys.stderr)

    xml_path = resolve_pick_place_scene(
        args.menagerie_path, explicit=args.menagerie_path is not None, spec=spec
    )
    mjm = load_pick_place_model(xml_path, spec)
    mjd = mujoco.MjData(mjm)
    apply_arm_ctrl(mjm, mjd, spec.home_ctrl)
    reset_cubes(mjm, mjd, spec)

    print(f"\n{spec.display_name}")
    print(f"{'nworld':>8} {'ms/step':>10} {'world-steps/s':>16} {'speedup vs 1':>14}")
    print("-" * 52)

    baseline = None
    for nworld in args.worlds:
        row = measure(mjm, mjd, nworld, nconmax=nconmax, njmax=njmax, steps=args.steps)
        baseline = baseline or row["world_steps_per_s"]
        print(
            f"{row['nworld']:>8} {row['per_step_ms']:>10.3f} "
            f"{row['world_steps_per_s']:>16,.0f} {row['world_steps_per_s'] / baseline:>13.1f}x"
        )

    print(
        "\nRead the ms/step column: it stays nearly flat while nworld grows by orders\n"
        "of magnitude. That flatness is the whole value proposition of MJWarp."
    )


if __name__ == "__main__":
    main()
