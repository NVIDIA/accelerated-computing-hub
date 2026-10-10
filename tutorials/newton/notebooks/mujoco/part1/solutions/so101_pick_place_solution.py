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
import json
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
)
from robots import add_robot_arg, get_robot  # noqa: E402
from utils import (  # noqa: E402
    print_asset_setup_help,
)

from box_task import BoxTask

SCRIPT_NAME = Path(__file__).name


def run_demo(
    xml_path: Path,
    spec,
    *,
    fps: int = 50,
    sim_substeps: int = 10,
    headless_steps: int = 0,
    debug: bool = False,
    box_task: BoxTask | None = None,
    report_path: Path | None = None,
) -> None:
    """Build the model, then run the pick-and-place loop on MuJoCo CPU."""
    # Step 1: compiled model + mutable state.
    model = load_pick_place_model(xml_path, spec)
    data = mujoco.MjData(model)

    # Step 2: grasp-ready home pose, cubes at their spawn positions.
    apply_arm_ctrl(model, data, spec.home_ctrl)
    reset_cubes(model, data, spec)

    # Step 3: the scripted controller (waypoints + IK + gradual gripper close).
    controller = box_task.make_controller() if box_task else PickPlaceController(spec=spec)
    # Initialize the arm above the first cube before any physics step.
    # The pending command is consumed once at frame0; payloads remain free.
    pending_phase = controller.phase_name() if box_task else None
    pending_ctrl = controller.step(model, data, 1.0 / fps).astype(np.float32) if box_task else None
    if box_task:
        apply_arm_ctrl(model, data, pending_ctrl)


    frame_dt = 1.0 / fps
    sim_dt = frame_dt / sim_substeps
    # Pin the physics clock to the controller rates (the SO-101 MJCF ships
    # timestep=0.005, which would silently desync controller and physics).
    model.opt.timestep = sim_dt
    if box_task:
        model.opt.iterations, model.opt.ls_iterations, model.opt.impratio = 100, 50, 100
        model.opt.tolerance = 1e-6  # Shared CPU/CUDA stopping target.

    print(f"Loaded: {xml_path}")
    print(f"Part 1 — MuJoCo CPU | {spec.display_name} | actuators={model.nu} | nq={model.nq} | dt={sim_dt:.5f}s")
    print("Sequence: pick red, then blue -> receiving box" if box_task else "Sequence: home -> pick red -> stack on blue")
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
        if box_task and frame_idx == 0:
            commanded_phase, ctrl = pending_phase, pending_ctrl
        else:
            commanded_phase = controller.phase_name()
            ctrl = controller.step(model, data, frame_dt)
        if box_task:
            ctrl = np.asarray(ctrl, dtype=np.float32)

        # Step 4: advance the physics — the line that defines Part 1.
        for _ in range(sim_substeps):
            data.ctrl[: model.nu] = ctrl
            mujoco.mj_step(model, data)

        if box_task:
            mujoco.mj_forward(model, data)
            box_task.observe(model, data, frame=frame_idx + 1, time=(frame_idx + 1) / fps, phase=commanded_phase)

        debug_frame()
        frame_idx += 1

    def report_completion() -> None:
        if box_task:
            report = box_task.report()
            report.update(backend='mujoco', device="cpu")
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            box_task.assert_complete()
            print(f"box check: PASS — both cubes grasped, lifted, carried, released and settled; {report['simulation_seconds']:.2f} simulated seconds")
            return
        # Step 5: report the final cube positions so the stack can be verified.
        red = data.xpos[red_body]
        blue = data.xpos[blue_body]
        xy_err = float(np.linalg.norm(red[:2] - blue[:2]))
        dz = float(red[2] - blue[2])
        print(f"Task complete. red={np.round(red, 3)} blue={np.round(blue, 3)}")
        print(f"stack check: xy_err={xy_err:.3f} m  dz={dz:.3f} m")
        return


    if headless_steps > 0:
        for _ in range(headless_steps):
            simulate_frame()
        report_completion()
        return

    # The interactive lesson has the same finite horizon as the headless example.
    # Rendering, pause and cancellation do not advance the controller clock.
    from viewer_loop import run_passive_frames

    complete = run_passive_frames(
        model, data, simulate_frame,
        frames=round((40 if box_task else 12) * fps), fps=fps,
        configure_viewer=lambda viewer: apply_viewer_camera(viewer, spec),
    )
    if complete:
        report_completion()


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1 solution — MuJoCo CPU pick & place.")
    add_robot_arg(parser)
    parser.add_argument("--task", choices=("stack", "box"), default="stack", help="Stack red on blue, or place both cubes in a receiving box.")
    parser.add_argument("--report", type=Path, default=None, help="Save measured --task box results as JSON.")
    parser.add_argument("--menagerie-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--headless-steps", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.report and args.task != "box":
        parser.error("--report requires --task box")
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

    run_demo(
        xml_path,
        spec,
        fps=args.fps,
        sim_substeps=args.sim_substeps,
        headless_steps=args.headless_steps,
        box_task=box_task,
        report_path=args.report,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
