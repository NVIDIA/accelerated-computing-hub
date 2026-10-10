#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 3 — SOLUTION: SO-101 pick & place on Newton's SolverMuJoCo
# ============================================================================
#
#  Steps 6-9 from so101_newton.py completed. Scene construction lives in
#  newton_scene_solution.py (Steps 0-5).

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco_warp as mjw
import warp as wp

import newton
import newton.examples

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pick_place_common as task  # noqa: E402
from newton_scene_solution import (  # noqa: E402
    body_index,
    build_ik_model,
    build_newton_model,
    robot_joint_indices,
    set_joint_targets,
)
from robots import get_robot  # noqa: E402


@wp.kernel
def _record_contact_peak(count: wp.array(dtype=wp.int32), peak: wp.array(dtype=wp.int32)):
    # Retain every substep's count, including inside a captured CUDA graph.
    wp.atomic_max(peak, 0, count[0])


class Example:
    """Pick-and-place on Newton's SolverMuJoCo backend (SO-101 or reBot DevArm)."""

    def __init__(self, viewer, args):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10  # 50 Hz control / 10 = 0.002 s physics (Article 2)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.spec = get_robot(getattr(args, "robot", None))

        self.model = build_newton_model(self.spec)
        self.red_body = body_index(self.model, "red_cube")
        self.blue_body = body_index(self.model, "blue_cube")

        # Step 6: Newton splits MjData into State / Control / Contacts. Two
        # states are needed because solver.step() is out-of-place.
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Native MuJoCo-C does NOT consume Newton Contacts in 1.6.0. Select
        # its own collision detection explicitly on CPU, for either robot.
        # CUDA uses MuJoCo Warp and keeps Newton contacts as the default;
        # --use-mujoco-contacts opts into MuJoCo detection on that path.
        use_mujoco_cpu = not self.model.device.is_cuda
        use_mujoco_contacts = use_mujoco_cpu or getattr(args, "use_mujoco_contacts", False)
        backend = "native MuJoCo-C (CPU)" if use_mujoco_cpu else "MuJoCo Warp (CUDA)"
        detection = "MuJoCo" if use_mujoco_contacts else "Newton"
        print(f"Physics: {backend}; contacts: {detection}; dt={self.sim_dt:.3f} s")

        # Step 7: SolverMuJoCo is the Newton interface to either engine.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=use_mujoco_cpu,  # CPU MuJoCo vs GPU MJWarp
            solver="newton",                # constraint solver: "newton" | "cg"
            integrator="implicitfast",      # "euler" | "rk4" | "implicitfast"
            # Newton's contact manifolds need more rows than Part 2's detector.
            njmax=max(self.spec.njmax, 1024),
            nconmax=self.spec.nconmax,
            cone="elliptic",                # friction cone: "pyramidal" | "elliptic"
            impratio=100,                   # normal-vs-friction impedance ratio
            iterations=100,                 # solver iterations per step
            ls_iterations=50,               # line-search iterations per solver iter
            use_mujoco_contacts=use_mujoco_contacts,
        )

        self.use_newton_contacts = not use_mujoco_contacts
        self.collision_pipeline = newton.CollisionPipeline(self.model) if self.use_newton_contacts else None
        self.contacts = self.collision_pipeline.contacts() if self.collision_pipeline is not None else None
        self.contact_peak = wp.zeros(1, dtype=wp.int32, device=self.model.device) if self.contacts is not None else None

        self.ik_model, self.ik_data, _ = build_ik_model(self.spec)
        joints = robot_joint_indices(self.model, self.ik_model)
        self.target_indices = self.model.joint_target_q_start.numpy()[joints]
        self.controller = task.PickPlaceController(spec=self.spec)

        self.viewer.set_model(self.model)
        self.viewer.picking_enabled = False
        self._frame_camera()

        # Step 9: capture the whole substep loop into a CUDA graph.
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _frame_camera(self):
        """Place Newton's camera close to the table, looking at the cubes."""
        if not hasattr(self.viewer, "set_camera"):
            return
        target = np.array(self.spec.camera_lookat, dtype=np.float64)
        cam_pos = target + np.array([0.18, -0.45, 0.30], dtype=np.float64)
        front = target - cam_pos
        front /= np.linalg.norm(front)
        yaw = float(np.degrees(np.arctan2(front[1], front[0])))
        pitch = float(np.degrees(np.arcsin(front[2])))
        self.viewer.set_camera(wp.vec3(*cam_pos), pitch, yaw)

    def simulate(self):
        # Step 8: out-of-place substep loop with a state swap.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
                wp.launch(_record_contact_peak, dim=1,
                          inputs=[self.contacts.rigid_contact_count, self.contact_peak],
                          device=self.model.device)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        ctrl = self.controller.step(self.ik_model, self.ik_data, self.frame_dt)
        set_joint_targets(self.control, ctrl, self.target_indices)

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Fail at frame boundaries under --test if MJWarp ran out of storage."""
        data = getattr(self.solver, "mjw_data", None)
        if data is None:  # Native MuJoCo-C does not use MJWarp buffers.
            return
        if self.contacts is not None:
            peak = int(self.contact_peak.numpy()[0])
            capacity = min(self.contacts.rigid_contact_max, data.naconmax)
            if peak > capacity:
                raise ValueError(
                    f"Newton contact capacity overflow: {peak} contacts exceed {capacity}; "
                    "increase the collision pipeline capacity and/or nconmax"
                )
        # The sticky flags survive all substeps in a captured CUDA graph.
        # Iteration limits are convergence warnings, not storage overflows.
        iteration_flags = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
        overflow = data.overflow.numpy() & ~iteration_flags
        if np.any(overflow):
            raise ValueError(
                f"MuJoCo Warp capacity overflow {overflow.tolist()}; "
                "increase njmax/nconmax before trusting the simulation"
            )

    def test_final(self):
        self.test_post_step()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
            raise ValueError("Nonfinite Newton body state")
        if not self.controller.done:
            raise ValueError("Pick-and-place sequence is incomplete; run 600 frames")
        red = body_q[self.red_body][:3]
        blue = body_q[self.blue_body][:3]
        xy_err = float(np.linalg.norm(red[:2] - blue[:2]))
        dz = float(red[2] - blue[2])
        print(f"red={np.round(red, 3)} blue={np.round(blue, 3)} xy_err={xy_err:.6f} dz={dz:.6f}")
        if xy_err > 0.015 or not (0.035 <= dz <= 0.055):
            raise ValueError("Red cube did not end stacked on the blue cube")
        print("Part 3 stack OK")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.add_argument(
            "--robot",
            default="so101",
            choices=("so101", "rebot"),
            help="Manipulator to load (default: so101).",
        )
        parser.add_argument("--task", choices=("stack", "box"), default="stack")
        parser.set_defaults(num_frames=600)
        return parser


def selected_task(argv=None):
    """Choose the example using the same task syntax as the full parsers."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--task", choices=("stack", "box"), default="stack")
    args, _ = parser.parse_known_args(argv)
    return args.task


if __name__ == "__main__":
    # Parse both --task box and --task=box before choosing the example.
    if selected_task() == "box":
        from box_newton_demo import main
        main()
        raise SystemExit(0)
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.viewer == "gl" and not args.headless:
        from viewer_loop import run_stack_gl
        if not run_stack_gl(viewer, lambda: Example(viewer, args), args):
            raise SystemExit(130)
    else:
        newton.examples.run(Example(viewer, args), args)
