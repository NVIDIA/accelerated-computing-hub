#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Physical sequential gripper pick/place of cubes, free cloth and a free cable."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import warp as wp
import newton
import newton.examples

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clean_table_scene_solution import build_scene, build_coupled_solver, make_clutter_pipeline
from clean_table_task import GraspEvidence, GripperIK, MIN_TOOL_CLEARANCE_M, check_mujoco_capacity
from newton_assets import NEWTON_ASSETS_REF


class Example:
    def __init__(self, viewer=None, args=None):
        self.args = args or SimpleNamespace(robot='so101', device='cpu')
        self.viewer = viewer
        self.fps, self.sim_substeps = 50, 10
        self.frame_dt, self.sim_dt = 0.02, 0.002
        self.frames, self.sim_time = 0, 0.
        self.scene = build_scene(getattr(self.args, 'robot', 'so101'), device=getattr(self.args, 'device', None))
        self.model = self.scene.model
        self.solver = build_coupled_solver(self.scene, vbd_iterations=getattr(self.args, 'vbd_iterations', 20),
                                           coupling_iterations=getattr(self.args, 'coupling_iterations', 2))
        self.pipeline = make_clutter_pipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.ctrl = self.scene.initial_ctrl.copy()
        self.drive_ctrl = self.ctrl.copy()
        self.ik = GripperIK(self.scene.ik_model, self.scene.spec)
        self.ik.ctrl = self.ctrl.copy()
        self.initial_points = self.scene.payload_points(self.state_0)
        self.evidence = {name: GraspEvidence(name, self.scene.bin, self.scene.table_z, points)
                         for name, points in self.initial_points.items()}
        self.targets = ('red_cube', 'blue_cube', 'cable', 'shirt')
        if getattr(self.args, 'target', None):
            self.targets = (self.args.target,)
        self.object_index, self.phase, self.phase_start = -1, 'initial_settle', 0.
        self.failure = None
        self.goal = self.scene.jaw_points(self.state_0).mean(axis=0)
        self.start = self.goal.copy()
        self.grip_start = self.grip_goal = self.scene.spec.gripper_open
        self.phase_duration = 1.
        self.closure = self.scene.spec.gripper_closed
        self.material = False
        self.release_roll_start = None
        self._frame_contacts = {name: np.zeros(2, dtype=np.int32) for name in self.initial_points}
        self._frame_forces = {name: np.zeros(2) for name in self.initial_points}
        self.max_coupling_force, self.max_soft_contacts, self.mujoco_warp_overflow_flags = 0., 0, 0
        self.history_body, self.history_particle, self.history_times = [], [], []
        self.history_phase, self.history_object, self.history_grip = [], [], []
        self.history_open, self.history_forces = [], []
        self.history_drive = []
        self.observations = []
        self._shape_body = self.model.shape_body.numpy()
        # Proxy contact buffers use destination-view indices. The VBD view is
        # compacted to the cable, jaws and cubes, so these are not global IDs.
        vbd_view = self.solver.view('deformable')
        global_bodies = {label: i for i, label in enumerate(self.model.body_label)}
        vbd_body_map = np.asarray([global_bodies[label] for label in vbd_view.body_label])
        local_shape_body = vbd_view.shape_body.numpy()
        self._vbd_shape_body = np.asarray([vbd_body_map[b] if b >= 0 else -1 for b in local_shape_body])
        self._mjc = self.solver.solver('rigid')
        self._geom_body = self._mjc.mj_model.geom_bodyid
        self._mjc_body_map = self._mjc.mjc_body_to_newton.numpy()[0]
        self._cube_bodies = {name: int(self._shape_body[shape]) for name, shape in self.scene.cube_shapes.items()}
        if self.model.device.is_cuda:
            self._contact_ids = wp.array(np.arange(self._mjc.mjw_data.naconmax), dtype=wp.int32, device=self.model.device)
            self._contact_force = wp.zeros(self._mjc.mjw_data.naconmax, dtype=wp.spatial_vector, device=self.model.device)
        self._observe()
        self._record_frame()
        if viewer is not None:
            viewer.set_model(self.model)
            viewer.picking_enabled = False
            if hasattr(viewer, 'set_camera'):
                viewer.set_camera(wp.vec3(self.scene.spec.red_xy[0] + 0.48, -0.68, self.scene.table_z + 0.56), -32., 133.)

    @property
    def active_object(self):
        return self.targets[self.object_index] if 0 <= self.object_index < len(self.targets) else ''

    def _set_phase(self, phase, goal, grip, duration):
        self.phase, self.phase_start, self.phase_duration = phase, self.sim_time, duration
        self.start = self.scene.jaw_points(self.state_0, material=self.material).mean(axis=0)
        self.goal = np.asarray(goal, dtype=float).copy()
        self.grip_start = float(self.ctrl[self.scene.spec.gripper_ctrl_indices[0]])
        self.grip_goal = float(grip)
        print(f'PHASE frame={self.frames} time={self.sim_time:.2f} object={self.active_object} phase={phase} '
              f'goal={np.round(self.goal,4).tolist()} grip={grip:.6f}', flush=True)

    def _next_object(self):
        self.object_index += 1
        if self.object_index >= len(self.targets):
            self._set_phase('final_settle', self.goal, self.scene.spec.gripper_open, 2.)
            return
        name = self.active_object
        self.release_roll_start = None
        self.material = name in ('shirt', 'cable')
        point = self.scene.grasp_point(self.state_0, name)
        self.grasp_goal = point.copy()
        if name == 'shirt':
            self.grasp_goal[2] -= 0.003
        elif name in self.scene.cube_shapes and self.scene.spec.key == 'rebot':
            self.grasp_goal[2] += 0.006
        self.closure = self.scene.spec.gripper_closed
        if self.material:
            # Calibrate actual native collision surfaces, not the cube aperture.
            self.ik.solve(point + [0,0,0.12], self.scene.spec.gripper_open, material=True, iterations=150)
            self.closure, measured = self.ik.closure_for_gap(0.011 if name == 'cable' else 0.003)
            print(f'CALIBRATION object={name} ctrl={self.closure:.6f} gap={measured:.6f}', flush=True)
            self.ik.ctrl = self.ctrl.copy()
        self.lift_goal = point.copy()
        self.lift_goal[2] = self.scene.table_z + (0.205 if self.material else 0.175)
        self._set_phase('approach', point + [0,0,0.10], self.scene.spec.gripper_open, 1.8)

    def _advance_phase(self):
        elapsed = self.sim_time - self.phase_start
        if elapsed + 1e-9 < self.phase_duration:
            return
        name, open_g = self.active_object, self.scene.spec.gripper_open
        ev = self.evidence.get(name)
        if self.phase == 'initial_settle':
            self._next_object()
        elif self.phase == 'approach':
            self.grasp_goal = self.scene.grasp_point(self.state_0, name)
            if name == 'shirt': self.grasp_goal[2] -= 0.003
            elif name in self.scene.cube_shapes and self.scene.spec.key == 'rebot': self.grasp_goal[2] += 0.006
            self._set_phase('descend', self.grasp_goal, open_g, 1.8)
        elif self.phase == 'descend':
            self._set_phase('close', self.grasp_goal, self.closure, 1.8)
        elif self.phase == 'close':
            if ev.grasped:
                self._set_phase('lift', self.lift_goal, self.closure, 2.)
            elif elapsed > self.phase_duration + 2.:
                self.failure = f'{name}: bilateral loaded grasp not acquired'
        elif self.phase == 'lift':
            if ev.lifted and not ev.dropped_before_release:
                target = self.lift_goal.copy(); target[:2] = self.scene.slots[name]
                self._set_phase('carry', target, self.closure, 3.)
            elif elapsed > self.phase_duration + 2.:
                self.failure = f'{name}: full-geometry loaded lift not sustained'
        elif self.phase == 'carry':
            self._set_phase('hold', self.goal, self.closure, 0.6)
        elif self.phase == 'hold':
            if ev.carried and not ev.dropped_before_release:
                turn_out_cloth = name == 'shirt' and self.scene.spec.key == 'so101'
                if turn_out_cloth:
                    self.release_roll_start = float(self.ctrl[self.scene.spec.wrist_lock_index])
                self._set_phase('open', self.goal, open_g, 2.0 if turn_out_cloth else 1.3)
            elif elapsed > self.phase_duration + 2.:
                self.failure = f'{name}: loaded whole-object carry did not reach receiving area'
        elif self.phase == 'open':
            if ev.released:
                self._set_phase('retreat', self.goal + [0,0,0.035], open_g, 1.2)
            elif elapsed > self.phase_duration + 2.:
                self.failure = f'{name}: jaw opening did not produce measured detachment'
        elif self.phase == 'retreat':
            self._set_phase('settle', self.goal, open_g, 1.3)
        elif self.phase == 'settle':
            if ev.report()['success']:
                self._next_object()
            elif elapsed > self.phase_duration + 4.:
                self.failure = f'{name}: released geometry did not settle entirely inside the bin'
        elif self.phase == 'final_settle':
            if all(self.evidence[n].report()['success'] for n in self.targets):
                self.phase = 'done'

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self._sample_contacts()

    def _sample_contacts(self):
        """Contact-native forces for the selected payload, accumulated over a frame."""
        axis_points = self.scene.jaw_points(self.state_0, material=self.material)
        axis = axis_points[1] - axis_points[0]
        axis /= max(np.linalg.norm(axis), 1e-9)
        vectors = {n: np.zeros((2,3)) for n in self.initial_points}
        counts = {n: np.zeros(2, dtype=np.int32) for n in self.initial_points}
        jaws = self.scene.jaw_bodies
        # MuJoCo owns rigid cube/jaw contacts, on CPU and CUDA.
        if self.model.device.is_cuda:
            import mujoco_warp as mjw
            data = self._mjc.mjw_data
            n = int(data.nacon.numpy()[0])
            mjw.contact_force(self._mjc.mjw_model, data, self._contact_ids, True, self._contact_force)
            geoms = data.contact.geom.numpy()[:n]
            forces = self._contact_force.numpy()[:n, :3]
        else:
            data = self._mjc.mj_data
            n = data.ncon
            geoms = np.asarray([c.geom for c in data.contact], dtype=int).reshape(-1,2)
            forces = []
            for i, contact in enumerate(data.contact):
                force = np.zeros(6); mujoco.mj_contactForce(self._mjc.mj_model, data, i, force)
                forces.append(contact.frame.reshape(3,3).T @ force[:3])
            forces = np.asarray(forces).reshape(-1,3)
        for pair, force in zip(geoms, forces):
            bodies = self._mjc_body_map[self._geom_body[pair]]
            for name, payload in self._cube_bodies.items():
                for j, jaw in enumerate(jaws):
                    if payload in bodies and jaw in bodies:
                        counts[name][j] += int(np.linalg.norm(force) > 1e-7)
                        vectors[name][j] += force if bodies[1] == jaw else -force
        # The VBD harvest already computes actual per-contact cable forces.
        contacts = self.solver.get_proxy_contacts('rigid', 'deformable')
        vbd = self.solver.solver('deformable')
        n = int(contacts.rigid_contact_count.numpy()[0])
        a, b = contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]
        force_array = contacts.rigid_contact_force
        rigid_reactions = np.zeros((2,3))
        if force_array is not None:
            cable = set(self.scene.cable_bodies)
            for sa, sb, force in zip(a,b,force_array.numpy()[:n]):
                bodies = (int(self._vbd_shape_body[sa]), int(self._vbd_shape_body[sb]))
                for j, jaw in enumerate(jaws):
                    if jaw in bodies:
                        reaction = force if bodies[1] == jaw else -force
                        rigid_reactions[j] += reaction
                        if any(body in cable for body in bodies):
                            counts['cable'][j] += int(np.linalg.norm(force)>1e-7)
                            vectors['cable'][j] += reaction
        raw = vbd.last_raw_body_f.numpy()
        vectors['shirt'] = raw[list(jaws), :3] - rigid_reactions
        ns = int(contacts.soft_contact_count.numpy()[0])
        for shape in contacts.soft_contact_shape.numpy()[:ns]:
            body = int(self._vbd_shape_body[shape])
            for j, jaw in enumerate(jaws):
                if body == jaw: counts['shirt'][j] += 1
        self.max_soft_contacts = max(self.max_soft_contacts, ns)
        self.max_coupling_force = max(self.max_coupling_force, float(np.linalg.norm(raw[:,:3])))
        for name in self.initial_points:
            normals = np.maximum(0., np.asarray([-np.dot(vectors[name][0],axis), np.dot(vectors[name][1],axis)]))
            self._frame_contacts[name] = np.maximum(self._frame_contacts[name], counts[name])
            # Keep a simultaneous pair from one real substep. Independently
            # maximizing each jaw could combine two unrelated unilateral touches.
            previous = self._frame_forces[name]
            if (float(normals.min()), float(normals.sum())) > (float(previous.min()), float(previous.sum())):
                self._frame_forces[name] = normals

    def step(self):
        if self.failure or self.phase == 'done': return
        self._advance_phase()
        if self.phase == 'done':
            self.observations[-1]['phase'] = self.phase
            self.observations[-1]['active_object'] = self.active_object
            return
        if self.failure: return
        elapsed = self.sim_time - self.phase_start
        alpha = np.clip(elapsed / max(self.phase_duration, 1e-9), 0., 1.)
        alpha = alpha * alpha * (3 - 2 * alpha)
        target = self.start + alpha * (self.goal - self.start)
        gripper = self.grip_start + alpha * (self.grip_goal - self.grip_start)
        release_roll = None
        if self.release_roll_start is not None and self.phase in ('open', 'retreat', 'settle', 'final_settle'):
            # Turning the fully opening hand lets gravity unwrap a free patch
            # draped over the fixed finger. This commands only the robot joint.
            blend = alpha if self.phase == 'open' else 1.
            release_roll = (1. - blend) * self.release_roll_start + blend * self.scene.spec.wrist_lock_value
        self.ctrl = self.ik.solve(target, gripper, material=self.material, release_roll=release_roll)
        if self.ik.last_position_error > 0.01:
            self.failure = f'{self.active_object}: IK target outside reachable workspace (residual {self.ik.last_position_error:.4f} m)'
            return
        self.drive_ctrl = self.ctrl.copy()
        if self.scene.spec.key == 'rebot':
            # Robot-only inverse dynamics supplies gravity feedforward through
            # the bounded position drives. Native actuator effort limits remain
            # active; no forces are applied to a payload.
            for actuator in range(self.scene.ik_model.nu):
                if actuator not in self.scene.spec.gripper_ctrl_indices:
                    joint = self.scene.ik_model.actuator_trnid[actuator, 0]
                    dof = self.scene.ik_model.jnt_dofadr[joint]
                    self.drive_ctrl[actuator] += self.ik.data.qfrc_bias[dof] / self.scene.spec.arm_target_ke
            self.drive_ctrl = np.clip(self.drive_ctrl, self.ik.lower, self.ik.upper)
        self.scene.set_targets(self.control, self.drive_ctrl)
        for name in self.initial_points:
            self._frame_contacts[name].fill(0); self._frame_forces[name].fill(0.)
        self.simulate()
        self.frames += 1; self.sim_time = self.frames * self.frame_dt
        self._observe()
        if self.frames % 5 == 0: self._record_frame()
        if getattr(self.args, 'debug', False) and self.frames % 25 == 0:
            name = self.active_object
            if name:
                print('OBS '+json.dumps(dict(object=name, phase=self.phase,
                    jaw_center=self.scene.jaw_points(self.state_0, material=self.material).mean(axis=0).tolist(),
                    goal=self.goal.tolist(), **self.evidence[name].last)), flush=True)
            else:
                print('SETTLE '+json.dumps(dict(frame=self.frames, time=self.sim_time, phase=self.phase,
                    tool_clearance_m=self.scene.tool_clearance(self.state_0),
                    objects={n: dict(inside=e.last['inside'], speed=e.last['max_point_speed'],
                                     settled=e.detached_settled_frames, success=e.report()['success'])
                             for n,e in self.evidence.items()})), flush=True)

    def _open_fraction(self):
        q = self.state_0.joint_q.numpy()[self.scene.robot_q_indices]
        actual = float(q[list(self.scene.spec.gripper_ctrl_indices)].mean())
        return float(np.clip((actual - self.closure) / max(self.scene.spec.gripper_open - self.closure, 1e-8), 0., 1.))

    def _observe(self):
        self.mujoco_warp_overflow_flags = check_mujoco_capacity(self._mjc)
        for name in ('body_q','body_qd','joint_q','joint_qd','particle_q','particle_qd'):
            if not np.isfinite(getattr(self.state_0, name).numpy()).all():
                raise ValueError(f'Nonfinite {name} at frame {self.frames}')
        speeds = self.scene.payload_speeds(self.state_0)
        clearance = self.scene.tool_clearance(self.state_0)
        for name, points in self.scene.payload_points(self.state_0).items():
            ev = self.evidence[name]
            ev.observe(phase=self.phase if name == self.active_object else 'inactive', time=self.sim_time,
                frame=self.frames, points=points, speed=speeds[name], jaw_contacts=self._frame_contacts[name],
                jaw_normal_forces=self._frame_forces[name], open_fraction=self._open_fraction(),
                hand_clear=clearance >= MIN_TOOL_CLEARANCE_M)
            if ev.dropped_before_release and not self.failure:
                self.failure = f'{name}: loaded bilateral grip lost before intentional release'
        self.observations.append(dict(frame=self.frames, time=self.sim_time, phase=self.phase,
            active_object=self.active_object, open_fraction=self._open_fraction(),
            contacts=[self._frame_contacts[n].copy() for n in self.initial_points],
            forces=[self._frame_forces[n].copy() for n in self.initial_points],
            lower=[self.evidence[n].last['bounds_min'] for n in self.initial_points],
            upper=[self.evidence[n].last['bounds_max'] for n in self.initial_points],
            centers=[self.evidence[n].last['center'] for n in self.initial_points],
            tool_clearance=clearance,
            speeds=[speeds[n] for n in self.initial_points]))

    def _record_frame(self):
        self.history_times.append(self.sim_time)
        self.history_body.append(self.state_0.body_q.numpy().copy())
        self.history_particle.append(self.state_0.particle_q.numpy().copy())
        self.history_phase.append(self.phase); self.history_object.append(self.active_object)
        self.history_grip.append(self.ctrl.copy()); self.history_open.append(self._open_fraction())
        self.history_drive.append(self.drive_ctrl.copy())
        self.history_forces.append(np.asarray([self._frame_forces[n] for n in self.initial_points]))

    def report(self):
        clearance = self.scene.tool_clearance(self.state_0)
        return dict(schema_version=2, newton=newton.__version__, warp=wp.__version__, device=str(self.model.device),
            robot=self.scene.spec.key, task='gripper_pick_place_into_bin' if len(self.targets)==4 else 'diagnostic_single_grasp',
            robot_asset_format='USD', newton_assets_ref=NEWTON_ASSETS_REF, frames=self.frames,
            simulation_seconds=self.sim_time, phase=self.phase, active_object=self.active_object, failure=self.failure,
            tool_withdrawn=bool(clearance >= MIN_TOOL_CLEARANCE_M), tool_clearance_m=clearance,
            max_coupling_input_force_norm=self.max_coupling_force, max_soft_contacts=self.max_soft_contacts,
            mujoco_warp_overflow_flags=self.mujoco_warp_overflow_flags, finite=True,
            bin_lower=self.scene.bin.lower.tolist(), bin_upper=self.scene.bin.upper.tolist(), table_z=self.scene.table_z,
            objects={name: ev.report() for name,ev in self.evidence.items()}, selected_objects=list(self.targets),
            ownership=dict(mujoco='robot, gripper and two cubes', vbd='cloth particles and cable bodies/joints'),
            material_geometry=dict(shirt='free single-layer shirt with an authored raised fold', cable_radius_m=0.007),
            coupling='SolverCoupledProxy (lagged, two-way, gripper joint proxies)')

    def is_complete(self):
        return bool(self.phase == 'done' and not self.failure and self.scene.tool_clearance(self.state_0) >= MIN_TOOL_CLEARANCE_M
                    and all(self.evidence[name].report()['success'] for name in self.targets))

    def test_final(self):
        report = self.report(); report['success'] = self.is_complete(); report['record_sha256'] = None
        record = getattr(self.args, 'record', None)
        if record:
            if self.history_times[-1] != self.sim_time: self._record_frame()
            else:
                self.history_phase[-1] = self.phase
                self.history_object[-1] = self.active_object
            path=Path(record); path=path if path.suffix=='.npz' else path.with_name(path.name+'.npz')
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,time=self.history_times,body_q=self.history_body,particle_q=self.history_particle,
                phase=np.asarray(self.history_phase,dtype='U32'),active_object=np.asarray(self.history_object,dtype='U16'),
                gripper_command=self.history_grip,jaw_open_fraction=self.history_open,jaw_normal_forces=self.history_forces,
                joint_drive_target=self.history_drive,
                observation_frame=np.asarray([o['frame'] for o in self.observations], dtype=np.int64),
                observation_time=np.asarray([o['time'] for o in self.observations]),
                observation_phase=np.asarray([o['phase'] for o in self.observations],dtype='U32'),
                observation_active_object=np.asarray([o['active_object'] for o in self.observations],dtype='U16'),
                observation_open_fraction=np.asarray([o['open_fraction'] for o in self.observations]),
                observation_jaw_contacts=np.asarray([o['contacts'] for o in self.observations],dtype=np.int32),
                observation_jaw_normal_forces=np.asarray([o['forces'] for o in self.observations]),
                observation_bounds_min=np.asarray([o['lower'] for o in self.observations]),
                observation_bounds_max=np.asarray([o['upper'] for o in self.observations]),
                observation_center=np.asarray([o['centers'] for o in self.observations]),
                observation_tool_clearance=np.asarray([o['tool_clearance'] for o in self.observations]),
                observation_max_point_speed=np.asarray([o['speeds'] for o in self.observations]),
                object_names=np.asarray(list(self.initial_points),dtype='U16'),
                shape_body=self.model.shape_body.numpy(),shape_transform=self.model.shape_transform.numpy(),
                shape_scale=self.model.shape_scale.numpy(),shape_type=self.model.shape_type.numpy(),
                shape_color=self.model.shape_color.numpy(),shape_label=np.asarray(self.model.shape_label,dtype='U256'),
                tri_indices=self.model.tri_indices.numpy(),joint_parent=self.model.joint_parent.numpy(),
                joint_child=self.model.joint_child.numpy(),bin_lower=self.scene.bin.lower,bin_upper=self.scene.bin.upper)
            report['record_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        if getattr(self.args,'report',None):
            path=Path(self.args.report);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(report,indent=2)+'\n')
        print('CLEAN_TABLE_RESULT '+json.dumps(report,sort_keys=True),flush=True)
        if not report['success']: raise ValueError(self.failure or 'Gripper task incomplete within the frame budget')
        print('Newton gripper pick/place OK: selected payloads grasped, carried, released and settled',flush=True)

    def render(self):
        if self.viewer:
            self.viewer.begin_frame(self.sim_time);self.viewer.log_state(self.state_0);self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser=newton.examples.create_parser()
        parser.add_argument('--robot',choices=('so101','rebot'),default='so101')
        parser.add_argument('--vbd-iterations',type=int,default=20)
        parser.add_argument('--coupling-iterations',type=int,default=2)
        parser.add_argument('--target',choices=('red_cube','blue_cube','shirt','cable'),help='Diagnostic single-object run; never a full-task receipt')
        parser.add_argument('--debug',action='store_true')
        parser.add_argument('--report');parser.add_argument('--record')
        parser.set_defaults(num_frames=5000)
        return parser


def run_checked(example, *, max_frames):
    if max_frames<=0: raise ValueError('The checked task needs a positive frame budget')
    try:
        while example.frames<max_frames and example.viewer.is_running() and not example.is_complete() and not example.failure:
            if example.phase == 'done':
                example.failure = 'Terminal phase lacks measured task evidence'
                break
            if example.viewer.should_step(): example.step()
            example.render()
        example.test_final()
    finally: example.viewer.close()


if __name__=='__main__':
    viewer,args=newton.examples.init(Example.create_parser())
    example=Example(viewer,args)
    if args.test: run_checked(example,max_frames=args.num_frames)
    else: newton.examples.run(example,args)
