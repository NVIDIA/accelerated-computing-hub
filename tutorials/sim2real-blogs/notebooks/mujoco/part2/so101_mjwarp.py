#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 2 — The same task on the GPU with MuJoCo Warp (MJWarp)
# ============================================================================
#
#  Identical scene, identical controller as Part 1. The ONLY thing that changes
#  is the physics backend: `mujoco.mj_step` becomes `mujoco_warp.step`, which
#  runs the MuJoCo pipeline on the GPU in float32 across a batch of worlds.
#
#  The Part 1 -> Part 2 migration, line for line:
#
#      MuJoCo CPU (Part 1)               MJWarp (Part 2)
#      ----------------------------      ---------------------------------------
#      model = load_pick_place_model()   mjm = load_pick_place_model()  (compile)
#                                        m   = mjw.put_model(mjm)       (upload)
#      data  = mujoco.MjData(model)      d   = mjw.make_data(mjm, nworld=...,
#                                                            nconmax=, njmax=)
#      data.ctrl[:] = targets            wp.copy(d.ctrl, wp.array(targets))
#      mujoco.mj_step(model, data)       mjw.step(m, d)
#      data.qpos                         d.qpos.numpy()                 (GPU->CPU)
#
#  Two modes live in this file:
#
#      migration-check mode (default, one world)
#          Mirrors world 0 into CPU MjData for rendering and task inspection.
#          Waypoint IK still uses its command-based scratch state on the CPU.
#          This checks migration behavior; it is not a throughput measurement.
#
#      throughput mode (--benchmark --nworld N)
#          Keeps everything on device, captures `mjw.step` into a CUDA graph and
#          measures steps/second. This is what MJWarp is actually for.
#
#  ---------------------------------------------------------------------------
#  YOUR TASK
#  ---------------------------------------------------------------------------
#  Search this file for "TODO Step" and complete each one, following
#  03__mujoco_warp.ipynb. Per-step snippets are in solutions/step_NN_*.py and
#  the finished file is solutions/so101_mjwarp_solution.py.
#
#  Run it:
#      python so101_mjwarp.py --headless-steps 600        # needs a CUDA GPU
#      python so101_mjwarp.py --benchmark --nworld 4096

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# TODO Step 0: import MuJoCo Warp and Warp itself.
# MJWarp is imported under the conventional short alias `mjw`.
#   import mujoco_warp as mjw
#   import warp as wp

from gpu_checks import check_mjwarp_state, require_cuda
from pick_place_common import (
    PickPlaceController,
    apply_arm_ctrl,
    apply_viewer_camera,
    load_pick_place_model,
    reset_cubes,
    resolve_pick_place_scene,
)
from robots import add_robot_arg, get_robot
from utils import (
    maybe_relaunch_with_mjpython,
    mjpython_viewer_error,
    print_asset_setup_help,
)

from box_task import BoxTask, assert_mjwarp_capacity

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
    box_task: BoxTask | None = None,
    report_path: Path | None = None,
) -> None:
    """Migration-check mode: one world on the GPU, mirrored back for the viewer."""
    nconmax = spec.nconmax if nconmax is None else nconmax
    njmax = spec.njmax if njmax is None else njmax
    device = require_cuda()
    print(f"Warp device: {device}")

    # Compile the MJCF on the host exactly like Part 1. We keep `mjd` (a CPU
    # MjData) around for two reasons: to seed the GPU state, and to mirror
    # world 0 back for rendering and task inspection.
    mjm = load_pick_place_model(xml_path, spec)
    # Pin the physics clock BEFORE put_model so the device copy inherits it.
    mjm.opt.timestep = (1.0 / fps) / sim_substeps
    mjd = mujoco.MjData(mjm)
    apply_arm_ctrl(mjm, mjd, spec.home_ctrl)
    reset_cubes(mjm, mjd, spec)

    # TODO Step 1: upload the compiled model to the GPU.
    # mjw.put_model() turns an mjModel into an mjw.Model whose fields are Warp
    # arrays living in device memory. It raises if the model uses a feature
    # MJWarp does not support yet — a much friendlier failure than silent drift.
    #   m = ...
    m = None

    # TODO Step 2: allocate batched device state.
    # mjw.make_data() needs three budgets that MuJoCo never asked you for:
    #   nworld  - how many worlds to simulate in parallel
    #   nconmax - expected contacts per world
    #   njmax   - maximum constraints per world (a hard limit)
    # Use nworld=1 here to compare the task outcome with Part 1.
    #   d = ...
    d = None

    # TODO Step 3: seed the device state from the CPU MjData, then run a
    # forward pass so derived quantities are valid before the first step.
    # MJWarp arrays are float32 and carry a leading world dimension, so a host
    # array of shape (nq,) becomes (1, nq) via mjd.qpos[None, :].
    # Copy qpos, qvel and ctrl, then call mjw.forward(m, d).

    controller = box_task.make_controller() if box_task else PickPlaceController(spec=spec)
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / sim_substeps

    print(f"Loaded: {xml_path}")
    print(f"Part 2 — MJWarp | {spec.display_name} | nworld=1 | IK on CPU, mjw.step on GPU | dt={sim_dt:.5f}s")

    frame_idx = 0

    def simulate_frame() -> None:
        nonlocal frame_idx
        commanded_phase = controller.phase_name()
        # Waypoint IK runs on a separate CPU scratch state derived from previous
        # commands. Mirrored device state supports rendering and task inspection;
        # this scripted controller does not use measured-state feedback.
        ctrl = controller.step(mjm, mjd, frame_dt)
        for _ in range(sim_substeps):
            mjd.ctrl[: mjm.nu] = ctrl

            # TODO Step 4: push the control vector to the device.
            # Wrap the host array with a leading world axis and wp.copy() it
            # into d.ctrl.

            # TODO Step 5: advance the physics on the GPU.
            # This single call replaces mujoco.mj_step from Part 1.

            if box_task:
                assert_mjwarp_capacity(d)

            # TODO Step 6: mirror world 0 to the host for rendering and task
            # inspection. d.qpos is a Warp array; .numpy() copies it back,
            # and index [0] selects world 0.

        frame_idx += 1
        if box_task:
            # Preserve GPU-solved contact forces when importing the observed state.
            mjw.forward(m, d)
            assert_mjwarp_capacity(d)
            mjw.get_data_into(mjd, mjm, d, world_id=0)
            box_task.observe(mjm, mjd, frame=frame_idx, time=frame_idx / fps, phase=commanded_phase)

    if headless_steps > 0:
        for _ in range(headless_steps):
            simulate_frame()
        diagnostics = check_mjwarp_state(d)
        print(f"Solver iteration-limit flags: {diagnostics or 'none'}")
        if box_task:
            report = box_task.report()
            report.update(backend='mujoco_warp', device=str(device))
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            box_task.assert_complete()
            print(f"box check: PASS — both cubes grasped, lifted, carried, released and settled; {report['simulation_seconds']:.2f} simulated seconds")
            return
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
            # Mirrored qpos/qvel require refreshed CPU kinematics for rendering.
            mujoco.mj_forward(mjm, mjd)
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
    """Time physics with fixed controls, excluding task/IK and transfers."""
    if nworld <= 0 or steps <= 0:
        raise ValueError("nworld and steps must be positive.")
    nconmax = spec.nconmax if nconmax is None else nconmax
    njmax = spec.njmax if njmax is None else njmax
    device = require_cuda()
    if not device.is_cuda:
        raise RuntimeError("GPU throughput measurements require an NVIDIA CUDA device.")

    mjm = load_pick_place_model(xml_path, spec)
    mjm.opt.timestep = 0.002
    mjd = mujoco.MjData(mjm)
    apply_arm_ctrl(mjm, mjd, spec.home_ctrl)
    reset_cubes(mjm, mjd, spec)

    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)

    # Broadcast the same initial state into every world.
    wp.copy(d.qpos, wp.array(np.tile(mjd.qpos, (nworld, 1)), dtype=wp.float32, device=device))
    wp.copy(d.qvel, wp.array(np.tile(mjd.qvel, (nworld, 1)), dtype=wp.float32, device=device))
    wp.copy(d.ctrl, wp.array(np.tile(mjd.ctrl, (nworld, 1)), dtype=wp.float32, device=device))
    mjw.forward(m, d)

    # TODO Step 7: capture mjw.step into a CUDA graph.
    # mjw.step is dozens of small kernel launches; replaying a captured graph
    # removes nearly all of that launch overhead. Capture once with
    # wp.ScopedCapture(), then replay with wp.capture_launch(graph).
    # This tutorial requires CUDA; capture the graph on the selected CUDA device.
    graph = None

    def advance() -> None:
        if graph is not None:
            wp.capture_launch(graph)
        else:
            mjw.step(m, d)

    for _ in range(10):  # warm-up: JIT compile and settle clocks
        advance()
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        advance()
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    diagnostics = check_mjwarp_state(d)  # Outside the timed region.
    print(f"Physics-only, fixed controls | timestep={mjm.opt.timestep:g}s | warmup=10")
    print(f"Solver iteration-limit flags: {diagnostics or 'none'}")
    total = steps * nworld
    print(f"nworld={nworld}  steps={steps}  graph={'yes' if graph else 'no'}")
    print(f"{elapsed:.3f} s for {total:,} world-steps")
    print(f"{total / elapsed:,.0f} world-steps/second")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2 — MJWarp pick & place.")
    add_robot_arg(parser)
    parser.add_argument("--task", choices=("stack", "box"), default="stack", help="Stack red on blue, or place both cubes in a receiving box.")
    parser.add_argument("--report", type=Path, default=None, help="Save measured --task box results as JSON.")
    parser.add_argument("--device", default=None, help="CUDA device, for example cuda:0 or cuda:1.")
    parser.add_argument("--menagerie-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--nconmax", type=int, default=None)
    parser.add_argument("--njmax", type=int, default=None)
    parser.add_argument("--headless-steps", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true", help="Run throughput mode.")
    parser.add_argument("--nworld", type=int, default=4096, help="Worlds for --benchmark.")
    parser.add_argument("--steps", type=int, default=200, help="Steps for --benchmark.")
    args = parser.parse_args()
    if args.report and args.task != "box":
        parser.error("--report requires --task box")
    if args.benchmark and args.task == "box":
        parser.error("The checked box task uses a single world; omit --benchmark")
    if args.device is not None:
        wp.set_device(args.device)
    spec = get_robot(args.robot)
    box_task = BoxTask(spec, fps=args.fps) if args.task == "box" else None
    if box_task:
        spec = box_task.spec

    try:
        xml_path = (box_task.resolve_scene(args.menagerie_path, explicit=args.menagerie_path is not None)
                    if box_task else resolve_pick_place_scene(
                        args.menagerie_path, explicit=args.menagerie_path is not None, spec=spec))
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
        box_task=box_task,
        report_path=args.report,
    )


if __name__ == "__main__":
    maybe_relaunch_with_mjpython()
    main()
