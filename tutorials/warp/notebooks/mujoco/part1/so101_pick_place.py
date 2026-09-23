#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 1 — Simulating the SO-101 with MuJoCo on the CPU
# ============================================================================
#
#  This is the reference implementation of the task: a manipulator
#  picks up a red cube and stacks it on a blue cube using a real friction grasp.
#  Default robot is the SO-101; pass --robot rebot for the Seeed reBot DevArm.
#  Everything runs on the CPU with stock MuJoCo, in double precision, in a
#  single world. It provides the reference task for the GPU migration in Part 2.
#
#  The three objects every MuJoCo program is built around:
#      mujoco.MjModel  — the compiled, read-only model (geometry, inertia, ...)
#      mujoco.MjData   — the mutable state (qpos, qvel, ctrl, time, contacts)
#      mujoco.mj_step  — advance the state by one model.opt.timestep
#
#  Per-frame loop in this script:
#      controller.step()  -> joint targets (Cartesian waypoint + IK)
#      data.ctrl[:] = targets
#      mj_step x sim_substeps
#      viewer.sync()
#
#  ---------------------------------------------------------------------------
#  YOUR TASK
#  ---------------------------------------------------------------------------
#  Search this file for "TODO Step" and complete each one, following the
#  instructions in 06__pick_and_place.ipynb. If you get stuck, the
#  snippet for each step is in solutions/step_NN_*.py and the finished file is
#  solutions/so101_pick_place_solution.py.
#
#  Run it:
#      python   so101_pick_place.py                    # Linux / Windows
#      mjpython so101_pick_place.py                    # macOS (passive viewer)
#      python   so101_pick_place.py --headless-steps 600 --debug
#      python   so101_pick_place.py --robot rebot --headless-steps 600

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# TODO Step 0: import the MuJoCo Python bindings.
# You need two modules: the core engine and the interactive viewer.
#   import mujoco
#   import mujoco.viewer

from pick_place_common import (
    PickPlaceController,
    apply_arm_ctrl,
    apply_viewer_camera,
    aperture_center,
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

SCRIPT_NAME = Path(__file__).name


def run_demo(
    xml_path: Path,
    spec,
    *,
    fps: int = 50,
    sim_substeps: int = 10,
    headless_steps: int = 0,
    debug: bool = False,
) -> None:
    """Build the model, then run the pick-and-place loop on MuJoCo CPU.

    Args:
        xml_path:       generated scene XML (robot + table + cubes).
        spec:           robot profile (SO-101 or reBot DevArm).
        fps:            render/control rate; one controller.step per frame.
        sim_substeps:   physics steps per frame (dt = 1/fps/substeps).
        headless_steps: if > 0, run N frames without a viewer (CI / timing).
        debug:          print gripper/cube telemetry on phase changes.
    """
    # TODO Step 1: compile the MJCF scene into an MjModel, then allocate the
    # mutable MjData that holds qpos / qvel / ctrl for this model.
    # Use load_pick_place_model(xml_path) rather than
    # mujoco.MjModel.from_xml_path() directly: it applies the shared arm force
    # boost so every part of the series simulates exactly the same model.
    #   model = ...
    #   data = ...
    model = None
    data = None

    # TODO Step 2: put the arm in the grasp-ready home pose and place the cubes
    # at their spawn positions. Two helpers from pick_place_common do this:
    # apply_arm_ctrl(model, data, spec.home_ctrl) and reset_cubes(model, data).

    # TODO Step 3: create the scripted controller that turns Cartesian
    # waypoints into joint targets (waypoints + IK + gradual gripper close).
    #   controller = PickPlaceController()
    controller = None

    frame_dt = 1.0 / fps
    sim_dt = frame_dt / sim_substeps
    # Pin the physics clock to the controller rates (the SO-101 MJCF ships
    # timestep=0.005, which would silently desync controller and physics).
    model.opt.timestep = sim_dt

    print(f"Loaded: {xml_path}")
    print(f"Part 1 — MuJoCo CPU | {spec.display_name} | actuators={model.nu} | nq={model.nq} | dt={sim_dt:.5f}s")
    print("Sequence: home -> pick red -> stack on blue")
    red_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    blue_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")
    frame_idx = 0
    last_phase = -1

    def debug_frame() -> None:
        nonlocal last_phase
        if not debug:
            return
        if controller.phase == last_phase and frame_idx % fps != 0:
            return
        last_phase = controller.phase
        print(
            f"[debug] frame={frame_idx:04d} phase={controller.phase}:{controller.phase_name()} "
            f"aperture={np.round(aperture_center(model, data), 3)} "
            f"red={np.round(data.xpos[red_body], 3)} "
            f"blue={np.round(data.xpos[blue_body], 3)} "
            f"ctrl={np.round(data.ctrl[: model.nu], 3)}"
        )

    def simulate_frame() -> None:
        # One control frame = one IK solve + several physics substeps.
        nonlocal frame_idx
        ctrl = controller.step(model, data, frame_dt)

        # TODO Step 4: advance the physics. For each of the sim_substeps:
        #   1. copy the controller output into data.ctrl[: model.nu]
        #   2. call mujoco.mj_step(model, data)
        # This is the line that defines Part 1 — in Part 2 it becomes
        # mjw.step(m, d).

        debug_frame()
        frame_idx += 1

    if headless_steps > 0:
        for _ in range(headless_steps):
            simulate_frame()
        # TODO Step 5: report the final cube positions so you can verify the
        # stack succeeded. Read data.xpos[red_body] and data.xpos[blue_body];
        # a good stack has matching XY and a Z gap of roughly 2 * 0.022 m.
        print("Headless complete.")
        return

    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as exc:
        raise mjpython_viewer_error(SCRIPT_NAME, exc) from exc

    with viewer_ctx as viewer:
        apply_viewer_camera(viewer, spec)
        print("MuJoCo viewer — Space = pause, Esc = quit.")
        while viewer.is_running():
            t0 = time.time()
            simulate_frame()
            viewer.sync()
            elapsed = time.time() - t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1 — MuJoCo CPU pick & place.")
    add_robot_arg(parser)
    parser.add_argument("--menagerie-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--headless-steps", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Print gripper/cube telemetry.")
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

    run_demo(
        xml_path,
        spec,
        fps=args.fps,
        sim_substeps=args.sim_substeps,
        headless_steps=args.headless_steps,
        debug=args.debug,
    )


if __name__ == "__main__":
    maybe_relaunch_with_mjpython()
    main()
