# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Shared geometry and state checks for the clean-table lesson.

These helpers read simulated state. They never reposition a payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import mujoco
import numpy as np
from scipy.optimize import least_squares
import warp as wp

import newton
import pick_place_common as task
from robots import RobotSpec


RIGID_CONTACTS_PER_BODY = 256
SOFT_CONTACTS_PER_BODY = 512
MIN_TOOL_CLEARANCE_M = 0.02


def check_mujoco_capacity(solver) -> int:
    """Reject sticky CUDA buffer overflows; retain convergence flags in reports."""
    if not hasattr(solver, "mjw_data"):
        return 0  # Native MuJoCo-C has no fixed Warp constraint buffers.
    import mujoco_warp as mjw

    flags = int(np.bitwise_or.reduce(solver.mjw_data.overflow.numpy(), initial=0))
    convergence = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
    if flags & ~convergence:
        raise ValueError(f"MuJoCo Warp capacity overflow (flags={flags}); increase the solver buffers")
    return flags


def validate_contact_counts(shape_bodies, rigid_pairs, soft_shapes, *,
                            rigid_limit=RIGID_CONTACTS_PER_BODY, soft_limit=SOFT_CONTACTS_PER_BODY):
    """Fail before VBD's fixed per-body lists truncate a contact stream."""
    for label, shapes, limit in (("rigid", np.asarray(rigid_pairs).reshape(-1), rigid_limit),
                                 ("soft", np.asarray(soft_shapes).reshape(-1), soft_limit)):
        if not len(shapes):
            continue
        bodies = np.asarray(shape_bodies)[shapes]
        bodies = bodies[bodies >= 0]  # The static world has no per-body VBD list.
        if len(bodies) and np.max(np.bincount(bodies)) > limit:
            raise ValueError(f"VBD per-body {label} contact capacity overflow (limit {limit})")


class CheckedCollisionPipeline(newton.CollisionPipeline):
    """Check both outer and proxy-local contacts before they enter a solver.

    Host-side diagnostics deliberately trade throughput for a fail-closed
    tutorial. This loop is not a CUDA-graph or performance benchmark.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.capacity_shape_bodies = model.shape_body.numpy()

    def collide(self, state, contacts):
        result = super().collide(state, contacts)
        nr = int(contacts.rigid_contact_count.numpy()[0])
        ns = int(contacts.soft_contact_count.numpy()[0])
        if nr > len(contacts.rigid_contact_shape0) or ns > len(contacts.soft_contact_shape):
            raise ValueError("Newton global contact capacity overflow")
        validate_contact_counts(
            self.capacity_shape_bodies,
            np.column_stack((contacts.rigid_contact_shape0.numpy()[:nr], contacts.rigid_contact_shape1.numpy()[:nr])),
            contacts.soft_contact_shape.numpy()[:ns],
        )
        return result


@dataclass(frozen=True)
class Bin:
    """The usable interior, measured at the inner faces of five solid walls."""

    lower: np.ndarray
    upper: np.ndarray
    tolerance: float = 0.003

    def contains(self, points: np.ndarray) -> bool:
        points = np.asarray(points)
        return bool(
            points.ndim == 2
            and points.shape[0] > 0
            and points.shape[1] == 3
            and np.isfinite(points).all()
            and np.all(points >= self.lower - self.tolerance)
            and np.all(points <= self.upper + self.tolerance)
        )


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:7])), dtype=float).reshape(3, 3)
    return np.asarray(points) @ rotation.T + pose[:3]


def expanded_points(points: np.ndarray, radius: float | np.ndarray) -> np.ndarray:
    """Axis extrema of spheres; sufficient for exact axis-aligned containment."""
    offsets = np.vstack((np.eye(3), -np.eye(3)))
    radii = np.broadcast_to(np.asarray(radius), (len(points),))
    return (np.asarray(points)[:, None, :] + radii[:, None, None] * offsets).reshape(-1, 3)


@dataclass
class GraspEvidence:
    """Measure one physical manipulation; observations never alter simulation state.

    ``observe`` accepts measured geometry/contact data so the success protocol can
    be tested independently of the physics. A lost accepted grasp is a failure;
    historical lift evidence never turns a later accidental drop into placement.
    """
    name: str
    bin: Bin
    table_z: float
    initial_points: np.ndarray
    fps: int = 50

    def __post_init__(self):
        self.initially_outside_bin = not self.bin.contains(self.initial_points)
        self.grasped = self.lifted = self.carried = False
        self.release_commanded = self.released = self.dropped_before_release = False
        self.grasp_time = self.lift_time = self.carry_time = None
        self.release_command_time = self.release_time = None
        self.max_loaded_bilateral_frames = self.full_lift_frames = self.carry_frames = 0
        self.detached_settled_frames = self.settled_frames = 0
        self._loaded_run = self._lift_run = 0
        self._last_frame = None
        self._lift_min = float('inf')
        self.min_lift_clearance_m = 0.0
        self.min_jaw_normal_force_N = float('inf')
        self.grasp_center = np.mean(self.initial_points, axis=0)
        self.carry_start_center = self.grasp_center.copy()
        self.carry_center = self.grasp_center.copy()
        self.carry_distance_m = 0.0
        self.carry_bounds_min = self.initial_points.min(axis=0).copy()
        self.carry_bounds_max = self.initial_points.max(axis=0).copy()
        self.over_bin_before_release = False
        self.release_open_fraction = 0.0
        self.attempts = 1
        self.last = {}

    def observe(self, *, phase, time, frame, points, speed, jaw_contacts,
                jaw_normal_forces, open_fraction, hand_clear=True):
        points = np.asarray(points, dtype=float)
        forces = np.asarray(jaw_normal_forces, dtype=float)
        contact_counts = np.asarray(jaw_contacts, dtype=float)
        contacts = contact_counts > 0
        finite = (np.isfinite(points).all() and np.isfinite(forces).all()
                  and np.isfinite(speed) and np.isfinite(open_fraction) and np.isfinite(time)
                  and np.isfinite(contact_counts).all())
        if (not finite or points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
                or forces.shape != (2,) or contacts.shape != (2,) or np.any(forces < 0)
                or np.any(contact_counts < 0) or np.any(contact_counts != np.floor(contact_counts))
                or type(frame) is not int or frame < 0 or time < 0 or speed < 0
                or not 0 <= open_fraction <= 1 or abs(time - frame / self.fps) > 1e-8
                or (self._last_frame is not None and frame != self._last_frame + 1)):
            raise ValueError(f'{self.name}: malformed or nonfinite manipulation observation')
        self._last_frame = frame
        center = points.mean(axis=0)
        lower, upper = points.min(axis=0), points.max(axis=0)
        clearance = float(lower[2] - self.table_z)
        loaded = bool(contacts.all() and np.all(forces > 1.0e-5))
        self._loaded_run = self._loaded_run + 1 if loaded else 0
        self.max_loaded_bilateral_frames = max(self.max_loaded_bilateral_frames, self._loaded_run)
        if phase == 'close' and self._loaded_run >= 10 and not self.grasped:
            self.grasped, self.grasp_time = True, float(time)
            self.grasp_center = center.copy()
        transporting = phase in ('lift', 'carry', 'hold')
        if self.grasped and transporting and not loaded and not self.release_commanded:
            self.dropped_before_release = True
        full_lift = loaded and clearance >= 0.01 and transporting and not self.dropped_before_release
        if full_lift:
            if self._lift_run == 0:
                self.carry_start_center = center.copy()
            self._lift_run += 1
            self._lift_min = min(self._lift_min, clearance)
            if self._lift_run > self.full_lift_frames:
                self.full_lift_frames = self._lift_run
                self.min_lift_clearance_m = self._lift_min
            if self.grasped and self._lift_run >= 25 and not self.lifted:
                self.lifted, self.lift_time = True, float(time)
        else:
            self._lift_run, self._lift_min = 0, float('inf')
            if transporting and not self.release_commanded:
                self.lifted = False
                self.lift_time = None
                self.full_lift_frames = 0
                self.min_lift_clearance_m = 0.
                self.carry_frames = 0
                self.carried = self.over_bin_before_release = False
                self.carry_time = None
                self.carry_distance_m = 0.
                self.min_jaw_normal_force_N = float('inf')
        over_bin = bool(np.all(lower[:2] >= self.bin.lower[:2] - self.bin.tolerance)
                        and np.all(upper[:2] <= self.bin.upper[:2] + self.bin.tolerance))
        if phase in ('carry', 'hold') and self.lifted and full_lift:
            self.carry_frames += 1
            self.min_jaw_normal_force_N = min(self.min_jaw_normal_force_N, float(forces.min()))
            distance = float(np.linalg.norm(center[:2] - self.carry_start_center[:2]))
            if self.carry_frames >= 10 and distance >= 0.05 and over_bin:
                self.carry_center = center.copy()
                self.carry_distance_m = distance
                self.carry_bounds_min, self.carry_bounds_max = lower.copy(), upper.copy()
                self.over_bin_before_release = True
                if not self.carried:
                    self.carried, self.carry_time = True, float(time)
        if phase == 'open' and not self.release_commanded:
            if not self.carried or self.dropped_before_release or not over_bin:
                raise ValueError(f'{self.name}: release requested without a retained loaded carry')
            self.release_commanded, self.release_command_time = True, float(time)
        if self.release_commanded and not self.released and not contacts.any() and open_fraction >= 0.8:
            self.released, self.release_time = True, float(time)
            self.release_open_fraction = float(open_fraction)
        inside = self.bin.contains(points)
        settled = self.released and not contacts.any() and inside and speed < 0.04 and hand_clear
        self.detached_settled_frames = self.detached_settled_frames + 1 if settled else 0
        self.settled_frames = self.detached_settled_frames
        self.last = dict(inside=inside, max_point_speed=float(speed), center=center.tolist(),
                         bounds_min=lower.tolist(), bounds_max=upper.tolist(),
                         jaw_contacts=contact_counts.astype(int).tolist(), jaw_normal_forces_N=forces.tolist(),
                         open_fraction=float(open_fraction), frame=int(frame), time=float(time),
                         current_lift_clearance_m=clearance)

    def report(self):
        values = dict(self.last)
        for name in ('initially_outside_bin', 'grasped', 'lifted', 'carried', 'release_commanded',
                     'released', 'dropped_before_release', 'grasp_time', 'lift_time', 'carry_time',
                     'release_command_time', 'release_time', 'max_loaded_bilateral_frames',
                     'full_lift_frames', 'carry_frames', 'min_lift_clearance_m', 'carry_distance_m',
                     'over_bin_before_release', 'release_open_fraction', 'detached_settled_frames',
                     'settled_frames', 'attempts'):
            values[name] = getattr(self, name)
        values['min_jaw_normal_force_N'] = (self.min_jaw_normal_force_N
                                           if np.isfinite(self.min_jaw_normal_force_N) else 0.0)
        for name in ('grasp_center', 'carry_start_center', 'carry_center', 'carry_bounds_min', 'carry_bounds_max'):
            values[name] = getattr(self, name).tolist()
        values['success'] = bool(self.grasped and self.lifted and self.carried and self.release_commanded
                                 and self.released and not self.dropped_before_release
                                 and self.detached_settled_frames >= self.fps and values.get('inside', False))
        return values


class GripperIK:
    """Position and horizontal closing-axis IK on a scratch MuJoCo model only."""
    def __init__(self, model, spec):
        self.model, self.spec = model, spec
        self.data = mujoco.MjData(model)
        self.ctrl = spec.home_ctrl.copy()
        self.lower, self.upper = task.actuator_ctrl_limits(model)
        self.body = model.body(spec.gripper_body).id
        self.free = list(spec.free_arm_joints)
        self.dofs = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in self.free]
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        self.material = False
        # Hinge fingertips converge and their connecting vector rotates sharply
        # near closure. Use a fixed hand-frame closing axis for orientation IK;
        # the physical fingertip positions still define the position objective.
        reference = self.ctrl.copy()
        reference[list(spec.gripper_ctrl_indices)] = -0.05 if spec.key == 'so101' else spec.gripper_closed
        task.apply_arm_ctrl(self.model, self.data, reference)
        self.material = True
        jaw_axis = self.jaw_points()[1] - self.jaw_points()[0]
        self.closing_axis_local = self.data.xmat[self.body].reshape(3, 3).T @ (jaw_axis / np.linalg.norm(jaw_axis))
        self.material = False
        task.apply_arm_ctrl(self.model, self.data, self.ctrl)

    def jaw_points(self, data=None):
        data = self.data if data is None else data
        if self.spec.key == 'so101':
            names = (('fixed_jaw_sph_tip1',), ('moving_jaw_sph_tip1',)) if self.material else (
                self.spec.left_jaw_geoms, self.spec.right_jaw_geoms)
            return np.asarray([task.geom_centroid(self.model, data, group) for group in names])
        points = []
        for name, sign in ((self.spec.left_jaw_body, -1), (self.spec.right_jaw_body, 1)):
            body = self.model.body(name).id
            offset = np.asarray([0., sign * 0.037, 0.]) if self.material else np.zeros(3)
            points.append(data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset)
        return np.asarray(points)

    def point(self):
        return self.jaw_points().mean(axis=0)

    def solve(self, target, gripper, *, material=False, iterations=30, release_roll=None):
        self.material = material
        ctrl = self.ctrl.copy()
        orient = material or self.spec.key == 'rebot'
        if self.spec.wrist_lock_index is not None:
            ctrl[self.spec.wrist_lock_index] = self.spec.wrist_lock_value - (np.pi if material and self.spec.key == 'so101' else 0.)
            if release_roll is not None:
                ctrl[self.spec.wrist_lock_index] = release_roll
        ctrl[list(self.spec.gripper_ctrl_indices)] = gripper
        if orient:
            free = self.free.copy()
            if material and self.spec.key == 'so101' and release_roll is None:
                free.append(self.spec.wrist_lock_index)
            lower, upper = self.lower[free].copy(), self.upper[free].copy()
            if material and self.spec.key == 'so101' and release_roll is None:
                reference_roll = self.spec.wrist_lock_value - np.pi
                lower[-1], upper[-1] = reference_roll - 0.3, reference_roll + 0.3
                ctrl[free[-1]] = np.clip(self.ctrl[free[-1]], lower[-1], upper[-1])
            use_tip_axis = self.spec.key == 'so101' and gripper >= -0.05
            def residual(values):
                candidate = ctrl.copy(); candidate[free] = values
                task.apply_arm_ctrl(self.model, self.data, candidate)
                jaws = self.jaw_points()
                axis = jaws[1] - jaws[0] if use_tip_axis else self.data.xmat[self.body].reshape(3, 3) @ self.closing_axis_local
                axis /= np.linalg.norm(axis)
                position = jaws.mean(axis=0) - target
                return np.r_[position, 0.05 * axis[2]] if release_roll is None else position
            seed = np.clip(ctrl[free], lower + 1e-9, upper - 1e-9)
            result = least_squares(residual, seed, bounds=(lower, upper),
                                   max_nfev=max(40, iterations), ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if np.linalg.norm(result.fun[:3]) > 0.003:
                alternatives = [self.spec.home_ctrl.copy()]
                if self.spec.key == 'so101':
                    for shoulder, elbow, wrist in ((-0.5, 0.5, 0.5), (0.3, -1.0, 1.1)):
                        alternative = ctrl.copy(); alternative[1:4] = (shoulder, elbow, wrist)
                        alternatives.append(alternative)
                for alternative in alternatives:
                    trial = least_squares(residual, np.clip(alternative[free], lower + 1e-9, upper - 1e-9),
                                          bounds=(lower, upper), max_nfev=100,
                                          ftol=1e-10, xtol=1e-10, gtol=1e-10)
                    if np.linalg.norm(trial.fun) < np.linalg.norm(result.fun):
                        result = trial
                    if np.linalg.norm(result.fun) < 3e-5:
                        break
            ctrl[free] = result.x
            self.ctrl = ctrl
            task.apply_arm_ctrl(self.model, self.data, ctrl)
            self.last_position_error = float(np.linalg.norm(np.asarray(target) - self.point()))
            return ctrl.copy()
        for _ in range(iterations):
            task.apply_arm_ctrl(self.model, self.data, ctrl)
            jaws = self.jaw_points()
            point = jaws.mean(axis=0)
            use_tip_axis = self.spec.key == 'so101' and gripper >= -0.05
            axis = (jaws[1] - jaws[0]) if use_tip_axis else self.data.xmat[self.body].reshape(3, 3) @ self.closing_axis_local
            axis /= np.linalg.norm(axis)
            position_error = np.asarray(target) - point
            error = np.r_[position_error, -0.05 * axis[2]] if orient else position_error
            if np.linalg.norm(error) < 3e-5:
                break
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, point, self.body)
            jac = self.jacp[:, self.dofs]
            if orient:
                axis_jac = np.cross(self.jacr[:, self.dofs].T, axis).T
                jac = np.vstack((jac, 0.05 * axis_jac[2]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(len(error)), error)
            delta = np.clip(0.6 * delta, -0.05, 0.05)
            # A clipped update at a joint limit must not increase the residual
            # and walk the arm into a remote, unreachable branch.
            accepted = False
            for scale in (1., 0.5, 0.25, 0.1, 0.03):
                candidate = ctrl.copy()
                candidate[self.free] += scale * delta
                candidate = np.clip(candidate, self.lower, self.upper)
                task.apply_arm_ctrl(self.model, self.data, candidate)
                trial = np.asarray(target) - self.jaw_points().mean(axis=0)
                if orient:
                    trial_axis = (self.jaw_points()[1] - self.jaw_points()[0]) if use_tip_axis else self.data.xmat[self.body].reshape(3, 3) @ self.closing_axis_local
                    trial_axis /= np.linalg.norm(trial_axis)
                    trial = np.r_[trial, -0.05 * trial_axis[2]]
                if np.linalg.norm(trial) < np.linalg.norm(error):
                    ctrl, accepted = candidate, True
                    break
            if not accepted:
                break
        self.ctrl = ctrl
        task.apply_arm_ctrl(self.model, self.data, ctrl)
        self.last_position_error = float(np.linalg.norm(np.asarray(target) - self.point()))
        return ctrl.copy()

    def closure_for_gap(self, gap):
        """Calibrate native jaw collision geometry using ray distances in scratch FK."""
        self.material = True
        index = self.spec.gripper_ctrl_indices[0]
        samples = []
        for q in np.linspace(self.lower[index], self.spec.gripper_open, 151):
            ctrl = self.ctrl.copy()
            ctrl[list(self.spec.gripper_ctrl_indices)] = q
            task.apply_arm_ctrl(self.model, self.data, ctrl)
            jaws = self.jaw_points()
            point = jaws.mean(axis=0)
            axis = jaws[1] - jaws[0]
            axis /= max(np.linalg.norm(axis), 1e-9)
            distances = []
            for sign in (-1, 1):
                geom = np.full(1, -1, dtype=np.int32)
                distances.append(mujoco.mj_ray(self.model, self.data, point, sign * axis,
                                             np.asarray([0, 0, 0, 1, 1, 0], dtype=np.uint8), 1, -1, geom))
            if min(distances) >= 0:
                samples.append((abs(sum(distances) - gap), q, sum(distances)))
        if not samples:
            raise ValueError('Could not calibrate physical gripper gap')
        _, q, actual = min(samples)
        return float(q), float(actual)


@dataclass
class CleanScene:
    model: newton.Model
    spec: RobotSpec
    bin: Bin
    bin_shapes: list[int]
    cube_shapes: dict[str, int]
    cable_bodies: list[int]
    cable_joints: list[int]
    cable_shapes: list[int]
    rigid_bodies: list[int]
    rigid_joints: list[int]
    clutter_bodies: list[int]
    clutter_joints: list[int]
    target_indices: np.ndarray
    robot_q_indices: np.ndarray
    initial_ctrl: np.ndarray
    ik_model: mujoco.MjModel
    ik_data: mujoco.MjData
    table_z: float
    jaw_bodies: tuple[int, int]
    jaw_shapes: tuple[list[int], list[int]]
    proxy_bodies: list[int]
    proxy_joints: list[int]
    grasp_sites: dict[str, tuple[tuple[int, np.ndarray], tuple[int, np.ndarray]]]
    slots: dict[str, np.ndarray]
    initial_grasp_points: dict[str, np.ndarray]

    def jaw_points(self, state, *, material=False):
        poses = state.body_q.numpy()
        return np.asarray([transform_points(poses[b], np.asarray([p]))[0]
                           for b, p in self.grasp_sites['material' if material else 'cube']])

    def tool_clearance(self, state):
        # Conservative transformed bounds include every collider on both jaws,
        # its local offset/rotation, and its contact margin. Model bounds already
        # contain shape scaling, including native mesh vertices.
        poses = state.body_q.numpy()
        local_pose = self.model.shape_transform.numpy()
        lower = self.model.shape_collision_aabb_lower.numpy()
        upper = self.model.shape_collision_aabb_upper.numpy()
        margins = self.model.shape_margin.numpy()
        bodies = self.model.shape_body.numpy()
        vertices = []
        for group in self.jaw_shapes:
            for shape in group:
                bounds = np.asarray(list(product(*zip(lower[shape] - margins[shape], upper[shape] + margins[shape]))))
                local = transform_points(local_pose[shape], bounds)
                vertices.append(transform_points(poses[bodies[shape]], local))
        return float(np.concatenate(vertices)[:, 2].min() - max(self.table_z, self.bin.upper[2]))

    def payload_points(self, state):
        body_q = state.body_q.numpy()
        shape_tf, scales, shape_body = (self.model.shape_transform.numpy(), self.model.shape_scale.numpy(),
                                        self.model.shape_body.numpy())
        corners = np.asarray(list(product((-1., 1.), repeat=3)))
        points = {}
        for name, shape in self.cube_shapes.items():
            local = transform_points(shape_tf[shape], corners * scales[shape])
            points[name] = transform_points(body_q[shape_body[shape]], local)
        cable = []
        for shape in self.cable_shapes:
            radius, half_height = scales[shape, :2]
            ends = np.asarray([[0., 0., -half_height], [0., 0., half_height]])
            local = transform_points(shape_tf[shape], ends)
            cable.append(expanded_points(transform_points(body_q[shape_body[shape]], local), radius))
        points['cable'] = np.concatenate(cable)
        points['shirt'] = expanded_points(state.particle_q.numpy(), self.model.particle_radius.numpy())
        return points

    def payload_speeds(self, state):
        velocity = state.body_qd.numpy()
        scales, shape_body = self.model.shape_scale.numpy(), self.model.shape_body.numpy()
        speeds = {}
        for name, shape in self.cube_shapes.items():
            v = velocity[shape_body[shape]]
            speeds[name] = float(np.linalg.norm(v[:3]) + np.linalg.norm(v[3:]) * np.linalg.norm(scales[shape]))
        speeds['cable'] = float(max(np.linalg.norm(velocity[shape_body[s], :3]) +
                                   np.linalg.norm(velocity[shape_body[s], 3:]) * sum(scales[s, :2])
                                   for s in self.cable_shapes))
        speeds['shirt'] = float(np.linalg.norm(state.particle_qd.numpy(), axis=1).max())
        return speeds

    def grasp_point(self, state, name):
        if name in self.cube_shapes:
            return self.payload_points(state)[name].mean(axis=0)
        if name == 'cable':
            poses = state.body_q.numpy()[self.cable_bodies]
            if self.spec.key == 'rebot':
                # Follow one interior material segment throughout the approach.
                # A settled rod's sub-micrometre height ties otherwise switch
                # the target to an end segment between approach and descent.
                return poses[(len(poses) - 1) // 2, :3].copy()
            return poses[np.argmax(poses[:, 2]), :3].copy()
        particles = state.particle_q.numpy()
        # A raised, freely moving fold is the authored grasp region.
        top = particles[particles[:, 2] > particles[:, 2].max() - 0.002]
        return top.mean(axis=0)

    def set_targets(self, control, ctrl):
        if np.shape(ctrl) != np.shape(self.target_indices) or not np.isfinite(ctrl).all():
            raise ValueError('Robot targets must be finite and match the actuator map')
        targets = control.joint_target_q.numpy()
        targets[self.target_indices] = ctrl
        control.joint_target_q.assign(targets)
