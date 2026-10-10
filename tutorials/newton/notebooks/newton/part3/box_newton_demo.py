# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Online two-cube pick-and-place through Newton's model/state/control API.

The model builder and physical observer are shared with the batched comparison.
This lesson computes IK online and can render every frame; it produces correctness
evidence, never a throughput result. The original stacking exercise is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import newton
import numpy as np
import warp as wp

import benchmark_box_protocol as protocol
import benchmark_newton_batch as runtime
from box_task import BoxTask
from robots import get_robot


class BoxExample:
    """One solver owns the state; only robot targets change during a rollout."""

    def __init__(self, robot, device, viewer):
        self.box = BoxTask(get_robot(robot))
        self.ik_model, self.ik_data, _ = runtime.build_ik_model(get_robot(robot), task_name="box")
        self.controller = self.box.make_controller()
        self.pending_phase = self.controller.phase_name()
        self.pending_ctrl = self.controller.step(self.ik_model, self.ik_data, .02).astype(np.float32)
        self.device, self.viewer = wp.get_device(device), viewer
        self.cpu = not self.device.is_cuda
        config = {"robot": robot, "backend": "newton_cpu" if self.cpu else "newton_cuda",
                  "nconmax": 64, "njmax": 128}
        # Reuse the comparison's builder/options, without its precomputed tape,
        # worker pool, sweep, timing or publication machinery.
        (self.model, self.solver, self.states, self.control, self.indices,
         _, _, self.metadata) = runtime._build(config, 1, self.pending_ctrl)
        self.targets = self.control.joint_target_q.numpy().copy()
        binding = runtime._newton_binding(self.solver.mj_model, self.box.spec)
        self.phases, self.commands = [], []
        self.frame_count, self.cpu_step_count = 0, 0
        self.native_finite = True
        self.history = np.empty((1, protocol.FRAMES, protocol.OBSERVATION_SIZE), dtype=np.float32)
        self.cursor = wp.zeros(1, dtype=int)
        self.graph = None
        if self.cpu:
            self.recorder = protocol.NativeRecorder(self.solver.mj_model, self.box.spec, **binding)
        else:
            self.data = self.solver.mjw_data
            self.recorder = protocol.CUDARecorder(self.solver.mj_model, self.solver.mjw_model,
                                                 self.data, self.box.spec, **binding)
            self.bad, self.peak_contacts = wp.zeros(1, dtype=int), wp.zeros(1, dtype=int)
            self.peak_constraints, self.step_counts = wp.zeros(1, dtype=int), wp.zeros(1, dtype=int)
            self.peak_broadphase = wp.zeros(1, dtype=int)
        self.reset_physics()
        if not self.cpu:
            self.simulate_frame()  # JIT before capture, then discard setup physics.
            self.reset_physics()
            with wp.ScopedCapture() as capture:
                self.simulate_frame()
            self.graph = capture.graph
            self.reset_physics()
        viewer.set_model(self.model)
        viewer.picking_enabled = False
        if hasattr(viewer, "set_camera"):
            target = np.asarray(self.box.spec.camera_lookat)
            position = target + [0.18, -0.45, 0.30]
            front = (target-position)/np.linalg.norm(target-position)
            viewer.set_camera(wp.vec3(*position), float(np.degrees(np.arcsin(front[2]))),
                              float(np.degrees(np.arctan2(front[1], front[0]))))

    def reset_physics(self):
        if self.cpu:
            mujoco.mj_resetData(self.solver.mj_model, self.solver.mj_data)
        else:
            mjw.reset_data(self.solver.mjw_model, self.data)
            for array in (self.bad, self.peak_contacts, self.peak_constraints,
                          self.step_counts, self.peak_broadphase):
                array.zero_()
        self.solver.reset(self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        self.cursor.zero_()

    def gpu_diagnostics(self, count_step):
        wp.launch(runtime._diagnostics, dim=(1, self.model.body_count), inputs=[
            self.states[0].body_q, self.states[0].body_qd, self.bad, self.model.body_count,
            self.data.nacon, self.data.nefc, self.peak_contacts, self.peak_constraints,
            self.step_counts, count_step, self.data.ncollision, self.peak_broadphase])

    def integrate_substep(self, state, other):
        state.clear_forces()
        self.solver.step(state, other, self.control, None, protocol.DT)
        return other, state

    def simulate_frame(self):
        state, other = self.states
        for _ in range(protocol.SUBSTEPS):
            state, other = self.integrate_substep(state, other)
            if self.cpu:
                self.cpu_step_count += 1
                self.native_finite &= bool(np.isfinite(self.solver.mj_data.qpos).all()
                                           and np.isfinite(self.solver.mj_data.qvel).all())
            else:
                # Record the current output buffer, including odd substeps.
                wp.launch(runtime._diagnostics, dim=(1, self.model.body_count), inputs=[
                    state.body_q, state.body_qd, self.bad, self.model.body_count,
                    self.data.nacon, self.data.nefc, self.peak_contacts, self.peak_constraints,
                    self.step_counts, 1, self.data.ncollision, self.peak_broadphase])
        if self.cpu:
            mujoco.mj_forward(self.solver.mj_model, self.solver.mj_data)
        else:
            mjw.forward(self.solver.mjw_model, self.data)
            self.gpu_diagnostics(0)
            self.recorder.record(self.cursor)
            wp.launch(runtime._advance, dim=1, inputs=[self.cursor])

    def step(self):
        if self.frame_count:
            phase = self.controller.phase_name()
            ctrl = self.controller.step(self.ik_model, self.ik_data, .02).astype(np.float32)
        else:
            phase, ctrl = self.pending_phase, self.pending_ctrl
        self.phases.append(phase)
        self.commands.append(ctrl.copy())
        self.targets[self.indices] = ctrl
        self.control.joint_target_q.assign(self.targets)
        if self.graph is None:
            self.simulate_frame()
        else:
            wp.capture_launch(self.graph)
        if self.cpu:
            self.recorder.capture(self.solver.mj_data, self.history[0, self.frame_count], self.frame_count)
        self.frame_count += 1

    def render(self):
        self.viewer.begin_frame(self.frame_count / 50)
        self.viewer.log_state(self.states[0])
        self.viewer.end_frame()

    def finish(self, report_path):
        wp.synchronize_device(self.device)
        warnings = {}
        if self.cpu:
            count = self.cpu_step_count
            warnings = {mujoco.mjtWarning(i).name: int(w.number)
                        for i,w in enumerate(self.solver.mj_data.warning) if w.number}
            capacity = self.native_finite and not warnings
            diagnostics = {"finite_all_substeps": self.native_finite, "warnings": warnings,
                           "recorded_physics_steps": count, "recorded_control_frames": self.frame_count}
            data = self.solver.mj_data
            qpos, qvel = data.qpos.copy(), data.qvel.copy()
        else:
            self.history = self.recorder.output.numpy()
            count = int(self.step_counts.numpy()[0])
            iteration_flags = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
            flags = self.data.overflow.numpy()
            warnings = {"overflow_flags": flags.tolist()}
            capacity = bool(not np.any(flags & ~iteration_flags) and not np.any(self.bad.numpy())
                            and int(self.peak_contacts.numpy()[0]) <= self.data.naconmax
                            and int(self.peak_constraints.numpy()[0]) <= self.data.njmax
                            and int(self.peak_broadphase.numpy()[0]) <= self.data.naconmax
                            and int(self.cursor.numpy()[0]) == protocol.FRAMES)
            diagnostics = {"peak_contacts": int(self.peak_contacts.numpy()[0]),
                           "peak_constraints": int(self.peak_constraints.numpy()[0]),
                           "peak_broadphase_pairs": int(self.peak_broadphase.numpy()[0]),
                           "naconmax": int(self.data.naconmax), "njmax": int(self.data.njmax),
                           "nonfinite_flags": self.bad.numpy().tolist(),
                           "recorded_control_frames": int(self.cursor.numpy()[0]),
                           "recorded_physics_steps": count,
                           "capacity_flags": (flags & ~iteration_flags).tolist(),
                           "iteration_flags": (flags & iteration_flags).tolist()}
            qpos, qvel = self.data.qpos.numpy()[0], self.data.qvel.numpy()[0]
        validation = protocol.validate_observations(self.history, self.box.spec, self.phases,
            clock_precision="float64" if self.cpu else "float32", step_counts=[count])
        trajectory = report_path.with_suffix('.npz')
        np.savez_compressed(trajectory, history=self.history, control_tape=np.asarray(self.commands),
                            phases=np.asarray(self.phases), final_qpos=qpos, final_qvel=qvel)
        report = {"status": "passed" if validation['passed'] and capacity else "failed",
                  "task": "two_cube_pick_place_into_box", "robot": self.box.spec.key,
                  "device": str(self.device), "frames": self.frame_count,
                  "recorded_physics_steps": count, "online_ik": True,
                  "publication_timing_eligible": False, "solver_metadata": self.metadata,
                  "capacity_passed": bool(capacity), "warnings": warnings, "diagnostics": diagnostics,
                  "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ("box_newton_demo.py", "benchmark_newton_batch.py", "benchmark_box_protocol.py",
                                  "box_task.py", "robots.py", "pick_place_common.py", "solutions/newton_scene_solution.py")},
                  "validation": validation,
                  "trajectory": trajectory.name, "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest()}
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        if report['status']!='passed':
            raise AssertionError(f"Physical box task failed; inspect {report_path}")
        print(f"PASS: {self.box.spec.key}, {self.device}, both cubes placed,40000 integration steps; {report_path}")


def main(example_type=BoxExample):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('box',),default='box')
    parser.add_argument('--robot',choices=('so101','rebot'),default='so101')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--viewer',choices=('null','gl','usd'),default='null')
    parser.add_argument('--num-frames',type=int,choices=(2000,),default=2000)
    parser.add_argument('--sim-substeps',type=int,choices=(20,),default=20)
    parser.add_argument('--test',action='store_true',help='Physical checks always run for this task.')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if any(args.report.with_suffix(suffix).exists() for suffix in ('.json', '.npz', '.usd')):
        raise FileExistsError('Preserve previous reports; choose a fresh report path.')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    wp.init();wp.set_device(args.device)
    if args.device!='cpu' and not wp.get_device().is_cuda:
        raise ValueError('Choose cpu or an available explicit CUDA device; no fallback.')
    from newton.viewer import ViewerNull, ViewerGL, ViewerUSD
    if args.viewer=='null':viewer=ViewerNull(num_frames=2000)
    elif args.viewer=='gl':viewer=ViewerGL()
    else:viewer=ViewerUSD(str(args.report.with_suffix('.usd')),fps=50,num_frames=2000)
    from viewer_loop import run_viewer_frames
    complete = run_viewer_frames(
        viewer, lambda: example_type(args.robot, args.device, viewer), frames=2000,
        render_fps=50 if args.viewer == 'gl' else None,
        finish=lambda example: example.finish(args.report),
    )
    if not complete:
        raise SystemExit(130)


if __name__=='__main__':
    main()
