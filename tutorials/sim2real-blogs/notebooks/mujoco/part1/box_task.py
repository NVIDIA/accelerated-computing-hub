# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Optional two-cube box task for MuJoCo and MuJoCo Warp.

This module only commands the robot. Free-cube coordinates are read during a
rollout, never written. The original stacking helpers and profiles are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import pick_place_common as task


NAMES = ("red_cube", "blue_cube")
CONTACT_FORCE_MIN = 1e-5


@dataclass(frozen=True)
class ReceivingBox:
    lower: np.ndarray
    upper: np.ndarray
    tolerance: float = 0.003

    def contains(self, points, *, xy_only=False):
        points = np.asarray(points, dtype=float)
        n = 2 if xy_only else 3
        return bool(points.ndim == 2 and points.shape[1] == 3 and len(points)
                    and np.isfinite(points).all()
                    and np.all(points[:, :n] >= self.lower[:n] - self.tolerance)
                    and np.all(points[:, :n] <= self.upper[:n] + self.tolerance))


class CubeEvidence:
    """Consecutive measured grasp/lift/carry/release windows, independent of IK."""

    def __init__(self, name, box, table_z, initial_points, fps=50):
        self.name, self.box, self.table_z, self.fps = name, box, table_z, fps
        self.initially_outside_bin = not box.contains(initial_points)
        self.grasped = self.lifted = self.carried = self.release_commanded = self.released = False
        self.dropped_before_release = False
        self.grasp_time = self.lift_time = self.carry_time = None
        self.release_command_time = self.release_time = None
        self.loaded_frames = self.max_loaded_bilateral_frames = self.full_lift_frames = 0
        self.carry_frames = self.detached_settled_frames = 0
        self.min_lift_clearance_m = self.carry_distance_m = self.release_open_fraction = 0.
        self.min_jaw_normal_force_N = float("inf")
        self.carry_start_center = np.mean(initial_points, axis=0)
        self.last_frame = None
        self.last = {}

    def observe(self, *, frame, time, phase, points, speed, counts, forces, opening):
        points = np.asarray(points, dtype=float)
        counts, forces = np.asarray(counts), np.asarray(forces, dtype=float)
        if (type(frame) is not int or frame < 1 or not np.isfinite(time)
                or abs(time - frame / self.fps) > 1e-8
                or (self.last_frame is not None and frame != self.last_frame + 1)
                or points.shape != (8, 3) or not np.isfinite(points).all()
                or counts.shape != (2,) or not np.isfinite(counts).all()
                or np.any(counts < 0) or np.any(counts != np.floor(counts))
                or forces.shape != (2,) or not np.isfinite(forces).all() or np.any(forces < 0)
                or not np.isfinite(speed) or speed < 0
                or not np.isfinite(opening) or not 0 <= opening <= 1):
            raise ValueError(f"{self.name}: invalid or nonconsecutive observation")
        self.last_frame = frame
        center = points.mean(axis=0)
        lower, upper = points.min(axis=0), points.max(axis=0)
        clearance = float(lower[2] - self.table_z)
        loaded = bool(np.all(counts > 0) and np.all(forces > CONTACT_FORCE_MIN))
        self.loaded_frames = self.loaded_frames + 1 if loaded else 0
        self.max_loaded_bilateral_frames = max(self.max_loaded_bilateral_frames, self.loaded_frames)
        if phase == "close" and self.loaded_frames >= 10 and not self.grasped:
            self.grasped, self.grasp_time = True, float(time)
        transporting = phase in ("lift", "carry", "hold")
        if transporting and self.grasped and not loaded and not self.release_commanded:
            self.dropped_before_release = True
        airborne = loaded and clearance >= 0.01 and transporting and not self.dropped_before_release
        if airborne:
            if self.full_lift_frames == 0:
                self.carry_start_center = center.copy()
                self.min_lift_clearance_m = clearance
            self.full_lift_frames += 1
            self.min_lift_clearance_m = min(self.min_lift_clearance_m, clearance)
            if self.grasped and self.full_lift_frames >= 25 and not self.lifted:
                self.lifted, self.lift_time = True, float(time)
        elif transporting and not self.release_commanded:
            self.full_lift_frames = self.carry_frames = 0
            self.lifted = self.carried = False
            self.lift_time = self.carry_time = None
            self.min_lift_clearance_m = self.carry_distance_m = 0.
            self.min_jaw_normal_force_N = float("inf")
        over_box = self.box.contains(points, xy_only=True)
        if phase in ("carry", "hold") and self.lifted and airborne:
            self.carry_frames += 1
            self.min_jaw_normal_force_N = min(self.min_jaw_normal_force_N, float(forces.min()))
            distance = float(np.linalg.norm(center[:2] - self.carry_start_center[:2]))
            if self.carry_frames >= 10 and distance >= .05 and over_box:
                self.carry_distance_m = distance
                if not self.carried:
                    self.carried, self.carry_time = True, float(time)
        if phase == "open" and not self.release_commanded:
            if not self.carried or self.dropped_before_release or not over_box:
                raise ValueError(f"{self.name}: opening without a retained whole-object carry")
            self.release_commanded, self.release_command_time = True, float(time)
        if self.release_commanded and not self.released and not np.any(counts) and opening >= .8:
            self.released, self.release_time = True, float(time)
            self.release_open_fraction = float(opening)
        inside = self.box.contains(points)
        settled = self.released and not np.any(counts) and inside and speed < .04
        self.detached_settled_frames = self.detached_settled_frames + 1 if settled else 0
        self.last = dict(frame=frame, time=float(time), bounds_min=lower.tolist(), bounds_max=upper.tolist(),
                         center=center.tolist(), inside=inside, max_point_speed=float(speed),
                         jaw_contacts=counts.astype(int).tolist(), jaw_normal_forces_N=forces.tolist(),
                         open_fraction=float(opening), whole_object_clearance_m=clearance)

    def report(self):
        result = dict(self.last)
        for key in ("initially_outside_bin", "grasped", "lifted", "carried", "release_commanded", "released",
                    "dropped_before_release", "grasp_time", "lift_time", "carry_time", "release_command_time",
                    "release_time", "max_loaded_bilateral_frames", "full_lift_frames", "carry_frames",
                    "min_lift_clearance_m", "carry_distance_m", "release_open_fraction", "detached_settled_frames"):
            result[key] = getattr(self, key)
        result["min_jaw_normal_force_N"] = (self.min_jaw_normal_force_N
                                             if np.isfinite(self.min_jaw_normal_force_N) else 0.)
        result["success"] = bool(self.initially_outside_bin and self.grasped and self.lifted and self.carried
                                  and self.release_commanded and self.released and not self.dropped_before_release
                                  and self.detached_settled_frames >= 50 and result.get("inside", False))
        return result


def native_contacts(model, data):
    """Read solved contact normal forces from the live native MuJoCo state."""
    result, force = [], np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, force)
        if not np.isfinite(force).all():
            raise ValueError("Nonfinite solved MuJoCo contact force")
        # get_data_into populates modern ``geom`` pairs. MuJoCo's deprecated
        # geom1/geom2 fields are separate storage and can retain stale IDs.
        first, second = map(int, contact.geom)
        result.append((first, second, max(0., float(force[0]))))
    return result


def assert_mjwarp_capacity(data):
    """Check raw device counts before MuJoCo Warp 3.8 can truncate a pull.

    Call after every physics substep (and a subsequent forward pass), before
    ``get_data_into``. Counts at capacity are valid; counts above it are not.
    """
    contacts = int(data.nacon.numpy()[0])
    constraints = np.asarray(data.nefc.numpy())
    if contacts < 0 or contacts > data.naconmax:
        raise ValueError(f"MuJoCo Warp contact capacity overflow: {contacts} > {data.naconmax}")
    if np.any(constraints < 0) or np.any(constraints > data.njmax):
        raise ValueError(f"MuJoCo Warp constraint capacity overflow: {constraints.tolist()} > {data.njmax}")
    # Newer releases also retain sticky flags. Iteration-limit warnings are
    # convergence diagnostics, not a contact/constraint storage overflow.
    flags = 0
    if hasattr(data, "overflow"):
        import mujoco_warp as mjw
        flags = int(np.bitwise_or.reduce(data.overflow.numpy(), initial=0))
        iteration_flags = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
        if flags & ~iteration_flags:
            raise ValueError(f"MuJoCo Warp capacity overflow: {flags}")
    return dict(contact_count=contacts, max_constraints=int(constraints.max(initial=0)), overflow_flags=flags)


def cube_points(model, data, name):
    geom = model.geom(name).id
    local = np.asarray(list(product((-1., 1.), repeat=3))) * model.geom_size[geom]
    return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def geom_lower_z(model, data, geom):
    """Exact primitive/mesh vertical support, including collider-local offsets."""
    kind = model.geom_type[geom]
    size, rotation, center = model.geom_size[geom], data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return float(np.min(model.mesh_vert[start:start + count] @ rotation[2] + center[2]))
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + abs(rotation[2, 2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[0] * np.linalg.norm(rotation[2, :2]) + abs(rotation[2, 2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.linalg.norm(rotation[2] * size)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(rotation[2]) @ size
    else:
        raise ValueError(f"Unsupported jaw collider type: {kind}")
    return float(center[2] - extent)


class BoxTask:
    """Scene, commands, and measured completion for an optional receiving box."""

    def __init__(self, spec, *, fps=50):
        if not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.base_spec = spec
        self.fps = fps
        x = .305 if spec.key == "so101" else .47
        # Use a task-specific profile copy so the stack and its migration retain
        # exactly their original layout. Both payloads begin outside the box.
        self.spec = replace(spec, table_center_xyz=(x, 0., spec.table_top_z - .012),
                            table_half_xyz=(.18, .28, .012), red_xy=(x - .035, -.15),
                            blue_xy=(x + .045, -.15), camera_lookat=(x, 0., spec.table_top_z + .11))
        self.box = ReceivingBox(np.array([x - .115, .055, spec.table_top_z + .006]),
                                np.array([x + .115, .245, spec.table_top_z + .091]))
        self.slots = {"red_cube": np.array([x - .065, .11]), "blue_cube": np.array([x + .005, .11])}
        self.evidence = {}
        self.frames = 0
        self.phase = "initial"
        self.tool_clearance = float("-inf")
        self._geom_ids = None
        self.sequence = self._sequence()

    def resolve_scene(self, menagerie_root=None, *, explicit=False):
        scene = task.resolve_pick_place_scene(menagerie_root, explicit=explicit, spec=self.base_spec)
        root = ET.fromstring(scene.read_text())
        root.set("model", self.spec.key + "_two_cube_box")
        world = root.find("worldbody")
        table = world.find("geom[@name='table']")
        table.set("pos", " ".join(map(str, self.spec.table_center_xyz)))
        table.set("size", " ".join(map(str, self.spec.table_half_xyz)))
        for name, position in zip(NAMES, (self.spec.red_cube_pos, self.spec.blue_cube_pos)):
            world.find(f"body[@name='{name}']").set("pos", " ".join(map(str, position)))
        lo, hi = self.box.lower, self.box.upper
        center, half, thickness = (lo + hi) / 2, (hi - lo) / 2, .006
        pieces = (
            ("floor", [center[0], center[1], lo[2] - thickness / 2], [half[0], half[1], thickness / 2]),
            ("left", [lo[0] - thickness / 2, center[1], center[2]], [thickness / 2, half[1] + thickness, half[2]]),
            ("right", [hi[0] + thickness / 2, center[1], center[2]], [thickness / 2, half[1] + thickness, half[2]]),
            ("near", [center[0], lo[1] - thickness / 2, center[2]], [half[0], thickness / 2, half[2]]),
            ("far", [center[0], hi[1] + thickness / 2, center[2]], [half[0], thickness / 2, half[2]]),
        )
        for name, pos, size in pieces:
            ET.SubElement(world, "geom", name="receiving_box_" + name, type="box",
                          pos=" ".join(map(str, pos)), size=" ".join(map(str, size)),
                          rgba="0.16 0.35 0.48 1", friction="0.8 0.005 0.0005", condim="3",
                          solref="0.01 1", contype="1", conaffinity="1")
        path = scene.with_name("scene_pick_place_box.xml")
        path.write_text(ET.tostring(root, encoding="unicode") + "\n")
        task.set_active_robot(self.spec)
        return path

    def _sequence(self):
        result = []
        spec = self.spec
        for name, spawn in zip(NAMES, (spec.red_cube_pos, spec.blue_cube_pos)):
            grasp = spawn + [0., 0., .006 if spec.key == "rebot" else 0.]
            above = grasp.copy(); above[2] = spec.table_top_z + .19
            destination = np.array([*self.slots[name], above[2]])
            retreat = destination + [0., 0., .025]
            for phase, point, grip, duration in (
                ("approach", above, spec.gripper_open, 1.6),
                ("descend", grasp, spec.gripper_open, 1.6),
                ("close", grasp, spec.gripper_closed, 2.2),
                ("lift", above, spec.gripper_closed, 2.0),
                ("carry", destination, spec.gripper_closed, 3.0),
                ("hold", destination, spec.gripper_closed, .8),
                ("open", destination, spec.gripper_open, 1.6),
                ("retreat", retreat, spec.gripper_open, 1.4),
                ("settle", retreat, spec.gripper_open, 1.4),
            ):
                result.append(task.Waypoint(name + ":" + phase, point, grip, duration))
        return result

    def make_controller(self):
        return task.PickPlaceController(sequence=self.sequence, spec=self.spec)

    def _bind(self, model):
        spec = self.spec
        if spec.left_jaw_geoms:
            roots = [int(model.geom_bodyid[model.geom(group[0]).id])
                     for group in (spec.left_jaw_geoms, spec.right_jaw_geoms)]
        else:
            roots = [model.body(spec.left_jaw_body).id, model.body(spec.right_jaw_body).id]
        groups = []
        for root in roots:
            bodies = {root}
            for body in range(root + 1, model.nbody):
                if model.body_parentid[body] in bodies:
                    bodies.add(body)
            groups.append({g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
                           and (model.geom_contype[g] or model.geom_conaffinity[g])})
        # SO-101's moving finger descends from its fixed-finger body. A contact
        # must belong to exactly one jaw before it can establish a bilateral grip.
        groups[0] -= groups[1]
        self._jaw_geoms = groups
        self._geom_ids = {name: model.geom(name).id for name in NAMES}

    def observe(self, model, data, *, frame, time, phase, contacts=None):
        """Observe the phase whose command advanced this frame, before transition.

        ``data`` must have current FK and velocities. In MJWarp mode pass actual
        `mujoco_warp.get_data_into` state, including its solved GPU contacts;
        do not run a native forward pass on that observation snapshot.
        """
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite MuJoCo state")
        if self._geom_ids is None:
            self._bind(model)
        if contacts is None:
            contacts = native_contacts(model, data)
        pairs = list(contacts)
        if any(len(c) != 3 or not np.isfinite(c[2]) or c[2] < 0 for c in pairs):
            raise ValueError("Invalid backend contact observations")
        actual = []
        for actuator in self.spec.gripper_ctrl_indices:
            joint = model.actuator_trnid[actuator, 0]
            actual.append(float(data.qpos[model.jnt_qposadr[joint]]))
        opening = float(np.clip((np.mean(actual) - self.spec.gripper_closed)
                                / (self.spec.gripper_open - self.spec.gripper_closed), 0., 1.))
        active, _, state = phase.partition(":")
        for name in NAMES:
            points = cube_points(model, data, name)
            if name not in self.evidence:
                self.evidence[name] = CubeEvidence(name, self.box, self.spec.table_top_z, points, self.fps)
            body = model.body(name).id
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            speeds = velocity[3:] + np.cross(velocity[:3], points - data.xpos[body])
            counts, forces = np.zeros(2, dtype=int), np.zeros(2)
            for first, second, force in pairs:
                if self._geom_ids[name] not in (first, second):
                    continue
                other = second if first == self._geom_ids[name] else first
                for jaw, geoms in enumerate(self._jaw_geoms):
                    if other in geoms:
                        counts[jaw] += 1
                        forces[jaw] += force
            self.evidence[name].observe(frame=frame, time=time, phase=state if name == active else "inactive",
                points=points, speed=float(np.max(np.linalg.norm(speeds, axis=1))),
                counts=counts, forces=forces, opening=opening)
        self.frames, self.phase = frame, phase
        self.tool_clearance = min(geom_lower_z(model, data, g) for geoms in self._jaw_geoms for g in geoms)
        self.tool_clearance -= max(self.spec.table_top_z, self.box.upper[2])

    def report(self):
        objects = {name: evidence.report() for name, evidence in self.evidence.items()}
        ordered = (len(objects) == 2 and all(objects[n]["success"] for n in NAMES)
                   and objects["blue_cube"]["grasp_time"] >= objects["red_cube"]["release_time"])
        return dict(schema_version=1, task="two_cube_pick_place_into_box", robot=self.spec.key,
                    frames=self.frames, simulation_seconds=self.frames / self.fps,
                    phase=self.phase, bin_lower=self.box.lower.tolist(), bin_upper=self.box.upper.tolist(),
                    table_z=self.spec.table_top_z, tool_clearance_m=(self.tool_clearance if np.isfinite(self.tool_clearance) else None),
                    objects=objects, success=bool(ordered and self.phase == "done" and self.tool_clearance >= .02))

    def assert_complete(self):
        report = self.report()
        if not report["success"]:
            detail = {n: {k: item[k] for k in ("grasped", "lifted", "carried", "released", "detached_settled_frames")}
                      for n, item in report["objects"].items()}
            raise AssertionError(f"Two-cube box task incomplete: phase={self.phase}, evidence={detail}, jaw clearance={self.tool_clearance:.4f}")
        return report
