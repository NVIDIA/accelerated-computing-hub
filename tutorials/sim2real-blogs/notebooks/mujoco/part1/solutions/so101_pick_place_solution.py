#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 1 — SOLUTION: SO-101 pick & place on MuJoCo CPU
# ============================================================================
#
#  Every "TODO Step" from so101_pick_place.py completed. Compare this against
#  your own file when you finish the exercises.

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Solution files live in solutions/, so make the part folder importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Step 0: import the MuJoCo Python bindings.
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from pick_place_common import (  # noqa: E402
    PickPlaceController,
    apply_arm_ctrl,
    apply_viewer_camera,
    aperture_center,
    load_pick_place_model,
    reset_cubes,
    resolve_pick_place_scene,
    validate_cpu_state,
    validate_stack,
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
    headless_steps: int = 0,
    debug: bool = False,
    test: bool = False,
) -> None:
    """Build the model, then run the pick-and-place loop on MuJoCo CPU."""
    if fps <= 0 or sim_substeps <= 0 or headless_steps < 0:
        raise ValueError("fps and sim_substeps must be positive; headless_steps must be non-negative.")
    if test and headless_steps <= 0:
        raise ValueError("--test requires --headless-steps greater than zero.")
    # Step 1: compiled model + mutable state.
    model = load_pick_place_model(xml_path, spec)
    data = mujoco.MjData(model)

    # Step 2: grasp-ready home pose, cubes at their spawn positions.
    apply_arm_ctrl(model, data, spec.home_ctrl)
    reset_cubes(model, data, spec)

    # Step 3: the scripted controller (waypoints + IK + gradual gripper close).
    controller = PickPlaceController(spec=spec)

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
        nonlocal frame_idx
        ctrl = controller.step(model, data, frame_dt)

        # Step 4: advance the physics — the line that defines Part 1.
        for _ in range(sim_substeps):
            data.ctrl[: model.nu] = ctrl
            mujoco.mj_step(model, data)

        debug_frame()
        frame_idx += 1

    if headless_steps > 0:
        for _ in range(headless_steps):
            simulate_frame()
        validate_cpu_state(data)
        mujoco.mj_forward(model, data)
        # Step 5: report the final cube positions so the stack can be verified.
        red = data.xpos[red_body]
        blue = data.xpos[blue_body]
        xy_err = float(np.linalg.norm(red[:2] - blue[:2]))
        dz = float(red[2] - blue[2])
        print(f"Headless complete. red={np.round(red, 3)} blue={np.round(blue, 3)}")
        print(f"stack check: xy_err={xy_err:.3f} m  dz={dz:.3f} m")
        if test:
            metrics = validate_stack(model, data, spec, controller_done=controller.done)
            print(f"Stack validation passed: {metrics}")
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
    parser = argparse.ArgumentParser(description="Part 1 solution — MuJoCo CPU pick & place.")
    add_robot_arg(parser)
    parser.add_argument("--menagerie-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--headless-steps", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true", help="Require a completed, settled stack in headless mode.")
    args = parser.parse_args()
    if args.test and args.headless_steps <= 0:
        parser.error("--test requires --headless-steps greater than zero")
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
        test=args.test,
    )


if __name__ == "__main__":
    maybe_relaunch_with_mjpython()
    main()
