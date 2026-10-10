# Optional earlier stacking exercise. The primary box exercise is box_newton_exercise.py.
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 3 — Building the scene the Newton way
# ============================================================================
#
#  Parts 1 and 2 handed an MJCF file straight to MuJoCo or MJWarp. Newton takes
#  a different route: you describe the scene with `newton.ModelBuilder`, which
#  can ingest MJCF, URDF and OpenUSD, and then `finalize()` it into a
#  `newton.Model` that any solver can consume.
#
#      Article 2     mujoco.MjModel.from_xml_path(...)   -> one format, one engine
#      Article 3     newton.ModelBuilder().add_mjcf(...) -> many formats, many solvers
#
#  What we deliberately REUSE from Article 2: the inverse kinematics and waypoint
#  controller in pick_place_common.py. IK is a kinematics helper and does not
#  care which dynamics solver runs underneath.
#
#  The same scene builder works for --robot so101 and --robot rebot.
#
#  ---------------------------------------------------------------------------
#  YOUR TASK
#  ---------------------------------------------------------------------------
#  Complete "TODO Step 0" through "TODO Step 5" here, then continue with
#  Steps 6-9 in so101_newton.py. Snippets are in solutions/step_NN_*.py.

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import warp as wp

# TODO Step 0: import Newton.
# You need three names: the top-level package, the joint target mode enum, and
# the MuJoCo solver class.
#   import newton
#   from newton import JointTargetMode
#   from newton.solvers import SolverMuJoCo

import pick_place_common as task
from robots import RobotSpec, get_robot

def build_ik_model(spec: RobotSpec | None = None) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    """Compile a stock MuJoCo model used *only* for inverse kinematics.

    The Part 1 controller solves IK with `mujoco.mj_jac`, so it needs a regular
    MjModel. This is pure kinematics on the CPU and is completely independent
    of the Newton dynamics model built below.
    """
    spec = spec or task.active_robot()
    scene = task.resolve_pick_place_scene(spec=spec)
    ik_model = mujoco.MjModel.from_xml_path(str(scene))
    ik_data = mujoco.MjData(ik_model)
    return ik_model, ik_data, scene


def build_newton_model(spec: RobotSpec | None = None):
    """Assemble the pick-and-place scene with ``newton.ModelBuilder``."""
    spec = spec or get_robot()
    scene = task.resolve_pick_place_scene(spec=spec)
    robot_xml = scene.parent / spec.robot_xml

    # TODO Step 1: select the Newton 1.5 coordinate target layout, create a
    # ModelBuilder, set the detection gap and register custom attributes.
    # SolverMuJoCo.register_custom_attributes(builder) MUST be called before
    # any asset is added: it teaches the builder about the MJWarp-specific
    # fields the solver needs to parse out of the MJCF.
    #   newton.use_coord_layout_targets = True
    #   builder = newton.ModelBuilder()
    #   builder.default_shape_cfg.gap = 0.0
    #   SolverMuJoCo.register_custom_attributes(builder)
    # gap=0 matches the MJCF; Newton 1.5 otherwise falls back to a 0.1 m gap,
    # which changes native MuJoCo contact generation at gripper scale.
    builder = None

    # TODO Step 2: import the robot from the same MJCF used in Article 2.
    # Call builder.add_mjcf() with the path to robot_xml. Pass
    # ignore_names=["floor"] (Newton adds its own ground plane below) and
    # collapse_fixed_joints=True (merges welded bodies, which keeps the
    # articulation smaller and the solver faster).

    # TODO Step 3: configure the joints as position-controlled actuators.
    # Compile robot_xml with mujoco.MjModel.from_xml_path() and use the
    # supplied robot_joint_indices(builder, robot_mj) helper to match names
    # in actuator order. Seed joint_q and joint_target_q at joint_q_start[j]
    # from spec.home_ctrl[actuator]. Gains/limits instead use joint_qd_start[j]:
    #   joint_target_ke    -> spec.arm_target_ke (stiffness)
    #   joint_target_kd    -> spec.arm_target_kd (damping)
    #   joint_target_mode  -> int(JointTargetMode.POSITION)
    #   joint_effort_limit -> spec.arm_effort_limit (skip gripper_ctrl_indices)
    # If spec.arm_force_limit is not None, also update the imported arm-only
    # builder.custom_attributes["mujoco:actuator_forcerange"].values[actuator].
    # Newton 1.5 preserves this independent force ceiling: changing only the
    # joint limit does NOT boost SO-101's actuators. Never boost the gripper.
    # reBot's actuator force ranges remain unchanged (arm_force_limit=None).

    # TODO Step 4: add the table and the two cubes.
    # The table is a static box: add_shape_box(body=-1, ...) where -1 means
    # "attached to the world". Each cube needs add_body() first (that is what
    # makes it dynamic) and then add_shape_box() on that body.
    # Use the same dimensions and poses as the MJCF so the task is unchanged:
    #   table  -> task.TABLE_HALF_SIZE at task.TABLE_CENTER
    #   cubes  -> task.CUBE_HALF at task.RED_CUBE_POS / task.BLUE_CUBE_POS
    # Suggested ShapeConfig values:
    #   table_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0e2, mu=1.0, density=0.0, gap=0.0)
    #   cube_cfg  = newton.ModelBuilder.ShapeConfig(ke=8.0e4, kd=8.0e2, mu=1.2, density=0.0, gap=0.0)
    # For each cube explicitly set add_body(mass=0.08, inertia=..., lock_inertia=True).
    # Its diagonal inertia is mass * (2 * task.CUBE_HALF)**2 / 6 on each axis.
    # Shape density=0 prevents adding a second mass; the positive BODY mass
    # makes the cube dynamic. The table is static because body=-1.

    # TODO Step 5: add a ground plane and finalize the builder into a Model.
    # finalize() packs everything into flat Warp arrays on the device. After
    # this the topology is fixed.
    #   builder.add_ground_plane()
    #   return builder.finalize()
    raise NotImplementedError("Complete TODO Steps 1-5 in build_newton_model()")


def body_index(model, suffix: str) -> int:
    """Return the body whose label ends with *suffix* (e.g. 'red_cube')."""
    matches = [i for i, label in enumerate(model.body_label) if label.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one body ending in '{suffix}', found {len(matches)}")
    return matches[0]


def robot_joint_indices(model, ik_model: mujoco.MjModel) -> np.ndarray:
    """Match scalar robot joints by name, in the IK actuator/control order.

    Accept a ModelBuilder or a finalized Model. Free payload joints are not
    servos; neither their coordinates nor their DOFs belong in this map.
    """
    joints = []
    for actuator in range(ik_model.nu):
        name = ik_model.joint(int(ik_model.actuator_trnid[actuator, 0])).name
        matches = [i for i, label in enumerate(model.joint_label)
                   if label == name or label.endswith('/' + name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one Newton joint for actuator joint {name!r}, got {matches}")
        joints.append(matches[0])
    return np.asarray(joints, dtype=np.int32)


def set_joint_targets(control, ctrl: np.ndarray, target_indices: np.ndarray) -> None:
    """Update only robot position targets, using Model.joint_target_q_start."""
    ctrl = np.asarray(ctrl)
    target_indices = np.asarray(target_indices)
    if ctrl.shape != target_indices.shape or ctrl.ndim != 1:
        raise ValueError("Controller output must match the robot target index vector")
    targets = control.joint_target_q.numpy()
    targets[target_indices] = ctrl
    control.joint_target_q.assign(targets)
