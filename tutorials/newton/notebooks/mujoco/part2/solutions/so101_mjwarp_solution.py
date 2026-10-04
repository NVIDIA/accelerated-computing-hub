#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 2 — SOLUTION: SO-101 pick & place on MJWarp
# ============================================================================
#
#  Every "TODO Step" from so101_mjwarp.py completed.

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Step 0: MuJoCo Warp and Warp.
import mujoco_warp as mjw  # noqa: E402
import warp as wp  # noqa: E402

from pick_place_common import (  # noqa: E402
    PickPlaceController,
    apply_arm_ctrl,
    apply_viewer_camera,
    load_pick_place_model,
    reset_cubes,
    resolve_pick_place_scene,
)
from robots import add_robot_arg, get_robot  # noqa: E402
from utils import (  # noqa: E402
    maybe_relaunch_with_mjpython,
    mjpython_viewer_error,
    print_asset_setup_help,
)

SCRIPT_NAME = Path(__file__).name


def run_demo(
    xml_path: Path,
    spec,
    *,
    fps: int = 50,
    sim_substeps: int = 10,
    nconmax: int | None = None,
    njmax: int | None = None,
    headless_steps: int = 0,
) -> None:
    """Parity mode: one world on the GPU, mirrored back for the viewer."""
    nconmax = spec.nconmax if nconmax is None else nconmax
    njmax = spec.njmax if njmax is None else njmax
    wp.init()
    device = wp.get_device()
    print(f"Warp device: {device}")

    mjm = load_pick_place_model(xml_path, spec)
    # Pin the physics clock BEFORE put_model so the device copy inherits it.
    mjm.opt.timestep = (1.0 / fps) / sim_substeps
    mjd = mujoco.MjData(mjm)
    apply_arm_ctrl(mjm, mjd, spec.home_ctrl)
    reset_cubes(mjm, mjd, spec)

    # Step 1: upload the compiled model to the device.
    m = mjw.put_model(mjm)

    # Step 2: allocate batched device state with explicit memory budgets.
    d = mjw.make_data(mjm, nworld=1, nconmax=nconmax, njmax=njmax)

    # Step 3: seed device state from the host, then run a forward pass.
    wp.copy(d.qpos, wp.array(mjd.qpos[None, :], dtype=wp.float32, device=device))
    wp.copy(d.qvel, wp.array(mjd.qvel[None, :], dtype=wp.float32, device=device))
    wp.copy(d.ctrl, wp.array(mjd.ctrl[None, :], dtype=wp.float32, device=device))
    mjw.forward(m, d)

    controller = PickPlaceController(spec=spec)
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / sim_substeps

    print(f"Loaded: {xml_path}")
    print(f"Part 2 — MJWarp | {spec.display_name} | nworld=1 | IK on CPU, mjw.step on GPU | dt={sim_dt:.5f}s")

    def simulate_frame() -> None:
        ctrl = controller.step(mjm, mjd, frame_dt)
        for _ in range(sim_substeps):
            mjd.ctrl[: mjm.nu] = ctrl

            # Step 4: host -> device.
            wp.copy(d.ctrl, wp.array(mjd.ctrl[None, :], dtype=wp.float32, device=device))

            # Step 5: the one line that defines Part 2.
            mjw.step(m, d)

            # Step 6: device -> host, mirroring world 0.
            mjd.qpos[:] = d.qpos.numpy()[0]
            mjd.qvel[:] = d.qvel.numpy()[0]

    if headless_steps > 0:
        for _ in range(headless_steps):
            simulate_frame()
        # qpos/qvel were mirrored from the device, but derived fields such as
        # xpos are stale until MuJoCo runs another forward pass on the host.
        mujoco.mj_forward(mjm, mjd)
        red = mjd.xpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "red_cube")]
        blue = mjd.xpos[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")]
        xy_err = float(np.linalg.norm(red[:2] - blue[:2]))
        dz = float(red[2] - blue[2])
        print(f"Headless complete. red={np.round(red, 3)} blue={np.round(blue, 3)}")
        print(f"stack check: xy_err={xy_err:.3f} m  dz={dz:.3f} m")
        return

    try:
        viewer_ctx = mujoco.viewer.launch_passive(mjm, mjd)
    except RuntimeError as exc:
        raise mjpython_viewer_error(SCRIPT_NAME, exc) from exc

    with viewer_ctx as viewer:
        apply_viewer_camera(viewer, spec)
        print("Viewer shows the MJWarp sim. Space = pause, Esc = quit.")
        while viewer.is_running():
            t0 = time.time()
            simulate_frame()
            viewer.sync()
            elapsed = time.time() - t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


def benchmark(
    xml_path: Path,
    spec,
    *,
    nworld: int = 4096,
    nconmax: int | None = None,
    njmax: int | None = None,
    steps: int = 200,
) -> None:
    """Throughput mode: many worlds, nothing leaves the GPU."""
    nconmax = spec.nconmax if nconmax is None else nconmax
    njmax = spec.njmax if njmax is None else njmax
    wp.init()
    device = wp.get_device()
    if not device.is_cuda:
        print("Benchmark needs a CUDA device; Warp is running on CPU.", file=sys.stderr)

    mjm = load_pick_place_model(xml_path, spec)
    mjd = mujoco.MjData(mjm)
    apply_arm_ctrl(mjm, mjd, spec.home_ctrl)
    reset_cubes(mjm, mjd, spec)

    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)

    wp.copy(d.qpos, wp.array(np.tile(mjd.qpos, (nworld, 1)), dtype=wp.float32, device=device))
    wp.copy(d.qvel, wp.array(np.tile(mjd.qvel, (nworld, 1)), dtype=wp.float32, device=device))
    wp.copy(d.ctrl, wp.array(np.tile(mjd.ctrl, (nworld, 1)), dtype=wp.float32, device=device))
    mjw.forward(m, d)

    # Step 7: capture the step into a CUDA graph and replay it.
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

    total = steps * nworld
    print(f"nworld={nworld}  steps={steps}  graph={'yes' if graph else 'no'}")
    print(f"{elapsed:.3f} s for {total:,} world-steps")
    print(f"{total / elapsed:,.0f} world-steps/second")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2 solution — MJWarp pick & place.")
    add_robot_arg(parser)
    parser.add_argument("--menagerie-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--headless-steps", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--nworld", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    spec = get_robot(args.robot)

    try:
        xml_path = resolve_pick_place_scene(
            args.menagerie_path, explicit=args.menagerie_path is not None, spec=spec
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error resolving assets: {exc}", file=sys.stderr)
        print_asset_setup_help(SCRIPT_NAME)
        sys.exit(1)

    if args.benchmark:
        benchmark(
            xml_path,
            spec,
            nworld=args.nworld,
            nconmax=args.nconmax,
            njmax=args.njmax,
            steps=args.steps,
        )
        return

    run_demo(
        xml_path,
        spec,
        fps=args.fps,
        sim_substeps=args.sim_substeps,
        nconmax=args.nconmax,
        njmax=args.njmax,
        headless_steps=args.headless_steps,
    )


if __name__ == "__main__":
    maybe_relaunch_with_mjpython()
    main()
