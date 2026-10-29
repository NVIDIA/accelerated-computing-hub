#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 1 — MuJoCo CPU exercise (intentionally incomplete)
# ============================================================================
#
#  Complete TODO Steps 0–4 to run the two-cube box task on the CPU.
#  Follow 02__pick_and_place.ipynb for the exercise instructions.
#  Default robot is the SO-101; pass --robot rebot for the Seeed reBot DevArm.
#
#  For a ready-to-run demo, use solutions/so101_pick_place_solution.py.
#  The solutions/step_NN_*.py files are reference snippets to read as needed;
#  they are not a sequence of programs to execute. Helper modules are imported
#  automatically. See ../../../README.md for setup and CPU commands.
#
#  Run from tutorials/sim2real-blogs/notebooks/mujoco with the environment active:
#      python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20
#      python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20 --headless-steps 2000
#
#  After completing the exercise:
#      python part1/so101_pick_place.py --task box --sim-substeps 20 --headless-steps 2000
#
#  The box episode lasts 40 simulated seconds. Pass --task stack to select the
#  earlier stacking exercise; its optional TODO Step 5 prints stack diagnostics.
#
#  Core objects: MjModel holds the compiled model, MjData holds mutable state,
#  and mujoco.mj_step advances physics by one model.opt.timestep.
#

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Ctrl+C policy first, so it also covers the slow imports and setup below.
from viewer_loop import EXIT_TASK_FAILED, install_ctrl_c, lesson_main
install_ctrl_c()

# TODO Step 0: import the MuJoCo Python bindings.
# You need two modules: the core engine and the interactive viewer.
#   import mujoco
#   import mujoco.viewer

from pick_place_common import (
    PickPlaceController,
    apply_arm_ctrl,
    apply_viewer_camera,
    aperture_center,
    check_stack,
    load_pick_place_model,
    reset_cubes,
    resolve_pick_place_scene,
)
from robots import add_robot_arg, get_robot
from utils import (
    print_asset_setup_help,
)

from box_task import BoxTask
from sim2real_shared.motion import MotionRecorder

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
    record_path: Path | None = None,
) -> int:
    """Build the model, then run the pick-and-place loop on MuJoCo CPU.

    Returns the process exit code: 0 task verified (or viewer quit), 1 task
    check failed, 130 Ctrl+C.

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
    #   controller = box_task.make_controller() if box_task else PickPlaceController(spec=spec)
    controller = None
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
        # One control frame = one IK solve + several physics substeps.
        nonlocal frame_idx
        if box_task and frame_idx == 0:
            commanded_phase, ctrl = pending_phase, pending_ctrl
        else:
            commanded_phase = controller.phase_name()
            ctrl = controller.step(model, data, frame_dt)
        if box_task:
            ctrl = np.asarray(ctrl, dtype=np.float32)

        # TODO Step 4: advance the physics. For each of the sim_substeps:
        #   1. copy the controller output into data.ctrl[: model.nu]
        #   2. call mujoco.mj_step(model, data)
        # This is the line that defines Part 1 — in Part 2 it becomes
        # mjw.step(m, d) and in Part 3 solver.step(...).

        if box_task:
            mujoco.mj_forward(model, data)
            box_task.observe(model, data, frame=frame_idx + 1, time=(frame_idx + 1) / fps, phase=commanded_phase)

        debug_frame()
        frame_idx += 1

    def report_completion() -> bool:
        if box_task:
            report = box_task.report()
            report.update(backend='mujoco', device="cpu")
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            try:
                box_task.assert_complete()
            except AssertionError as exc:
                print(f"Task FAILED: box check did not pass: {exc}")
                return False
            print(f"box check: PASS — both cubes grasped, lifted, carried, released and settled; {report['simulation_seconds']:.2f} simulated seconds")
            print("Task complete: both cubes are in the receiving box (verified).")
            return True
        # TODO Step 5: report the final cube positions so you can verify the
        # stack succeeded. Read data.xpos[red_body] and data.xpos[blue_body];
        # a good stack has matching XY and a Z gap of roughly 2 * 0.022 m.
        print("Final cube positions: (complete TODO Step 5)")
        mujoco.mj_forward(model, data)
        passed, summary = check_stack(model, data, spec)
        if passed:
            print(f"Task complete: red cube stacked on blue (verified: {summary}).")
        else:
            print(f"Task FAILED: red cube is not stacked on blue ({summary}).")
        return passed


    if headless_steps > 0:
        recorder = (MotionRecorder(record_path, data, fps=fps, xml_path=xml_path,
                                   robot=spec.key, backend="mujoco") if record_path else None)
        for _ in range(headless_steps):
            simulate_frame()
            if recorder:
                recorder.capture(data, frame_idx)
        passed = report_completion()
        if recorder:
            recorder.save(success=passed)
        return 0 if passed else EXIT_TASK_FAILED

    # The interactive lesson has the same finite horizon as the headless example.
    # Rendering, pause and cancellation do not advance the controller clock.
    from viewer_loop import run_passive_frames

    result = run_passive_frames(
        model, data, simulate_frame,
        frames=round((40 if box_task else 12) * fps), fps=fps,
        configure_viewer=lambda viewer: apply_viewer_camera(viewer, spec),
    )
    if not result.complete:
        return result.exit_code
    return 0 if report_completion() else EXIT_TASK_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 1 — MuJoCo CPU pick & place.")
    add_robot_arg(parser)
    parser.add_argument("--task", choices=("stack", "box"), default="stack", help="Stack red on blue, or place both cubes in a receiving box.")
    parser.add_argument("--report", type=Path, default=None, help="Save measured --task box results as JSON.")
    parser.add_argument("--record", type=Path, help="Save sampled poses from a headless box episode for inline replay.")
    parser.add_argument("--menagerie-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--headless-steps", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Print gripper/cube telemetry.")
    args = parser.parse_args()
    if args.report and args.task != "box":
        parser.error("--report requires --task box")
    if args.record and (args.task != "box" or args.headless_steps <= 0):
        parser.error("--record requires --task box and --headless-steps")
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

    return run_demo(
        xml_path,
        spec,
        fps=args.fps,
        sim_substeps=args.sim_substeps,
        headless_steps=args.headless_steps,
        box_task=box_task,
        report_path=args.report,
        record_path=args.record,
        debug=args.debug,
    )


if __name__ == "__main__":
    lesson_main(main)
