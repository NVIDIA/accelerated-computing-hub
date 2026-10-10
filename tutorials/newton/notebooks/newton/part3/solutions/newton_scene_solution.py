# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# ============================================================================
#  Part 3 — SOLUTION: building the scene with newton.ModelBuilder
# ============================================================================
#
#  Steps 0-5 from newton_scene.py completed.

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Step 0: import Newton.
import newton  # noqa: E402
from newton import JointTargetMode  # noqa: E402
from newton.solvers import SolverMuJoCo  # noqa: E402

import pick_place_common as task  # noqa: E402
from robots import RobotSpec, get_robot  # noqa: E402


def build_ik_model(spec: RobotSpec | None = None, *, task_name: str = "stack") -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    """Compile a stock MuJoCo model used *only* for inverse kinematics."""
    spec = spec or task.active_robot()
    if task_name == "box":
        from box_task import BoxTask
        scene = BoxTask(spec).resolve_scene()
    elif task_name == "stack":
        scene = task.resolve_pick_place_scene(spec=spec)
    else:
        raise ValueError(f"Unknown task {task_name!r}")
    ik_model = mujoco.MjModel.from_xml_path(str(scene))
    ik_data = mujoco.MjData(ik_model)
    return ik_model, ik_data, scene


def build_newton_builder(spec: RobotSpec | None = None, *, task_name: str = "stack",
                         initial_ctrl: np.ndarray | None = None) -> newton.ModelBuilder:
    """Assemble one complete scene, ready to finalize or replicate into worlds."""
    spec = spec or get_robot()
    if task_name == "box":
        from box_task import BoxTask
        box_task = BoxTask(spec)
        scene = box_task.resolve_scene()
        spec = box_task.spec
    elif task_name == "stack":
        scene = task.resolve_pick_place_scene(spec=spec)
    else:
        raise ValueError(f"Unknown task {task_name!r}")
    robot_xml = scene.parent / spec.robot_xml

    # Step 1: explicit Newton 1.5 coordinate layout + solver attributes.
    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    # Match the MJCF's zero detection gap, not Newton 1.5's 0.1 m fallback.
    # The latter changes native MuJoCo contact generation at gripper scale.
    builder.default_shape_cfg.gap = 0.0
    SolverMuJoCo.register_custom_attributes(builder)

    # Step 2: import the robot from the same MJCF used in Article 2.
    builder.add_mjcf(
        str(scene if task_name == "box" else robot_xml),
        ignore_names=["floor"],
        # Keep authored static robot bodies in the box benchmark so MuJoCo's
        # parent/body contact exclusions survive conversion into a template.
        collapse_fixed_joints=task_name != "box",
    )

    # Step 3: seed only named robot joints, in the MJCF actuator order.
    # Builder targets use coordinates; gains and effort limits use DOFs.
    robot_mj = mujoco.MjModel.from_xml_path(str(robot_xml))
    joints = robot_joint_indices(builder, robot_mj)
    seed = np.asarray(spec.home_ctrl if initial_ctrl is None else initial_ctrl, dtype=float)
    if seed.shape != (robot_mj.nu,) or not np.isfinite(seed).all():
        raise ValueError("Initial robot coordinates must match all finite actuator targets")
    gripper = set(spec.gripper_ctrl_indices)
    for actuator, joint in enumerate(joints):
        q = builder.joint_q_start[joint]
        dof = builder.joint_qd_start[joint]
        builder.joint_q[q] = float(seed[actuator])
        builder.joint_target_q[q] = float(seed[actuator])
        builder.joint_target_ke[dof] = spec.arm_target_ke
        builder.joint_target_kd[dof] = spec.arm_target_kd
        builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
        if actuator in gripper:
            if task_name == "box" and spec.key == "rebot":
                # Match the authored linear-finger drives, as in the coupled
                # box example; rotary-arm gains make this grasp too soft.
                builder.joint_target_ke[dof] = float(robot_mj.actuator_gainprm[actuator, 0])
                builder.joint_target_kd[dof] = float(-robot_mj.actuator_biasprm[actuator, 2])
            continue
        builder.joint_effort_limit[dof] = spec.arm_effort_limit
        # Newton 1.5 also preserves the MJCF actuator's own force range.
        # Raising only joint_effort_limit leaves SO-101 capped at 2.94 Nm.
        # Mirror Article 2's arm-only boost; leave both robots' grippers and
        # reBot's heterogeneous actuator limits exactly as imported.
        if spec.arm_force_limit is not None:
            builder.custom_attributes["mujoco:actuator_forcerange"].values[actuator] = wp.vec2(
                -spec.arm_force_limit, spec.arm_force_limit,
            )

    if task_name == "box":
        # The benchmark migrates the complete canonical MJCF scene, including
        # table, free payloads, box geometry and their authored contact masks.
        # The tutorial's programmatic stacking scene continues below.
        builder.add_ground_plane()
        return builder

    # Step 4: table (static) and the two cubes (free bodies), same poses as the MJCF.
    table_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0e2, mu=1.0, density=0.0, gap=0.0)
    cube_cfg = newton.ModelBuilder.ShapeConfig(ke=8.0e4, kd=8.0e2, mu=1.2, density=0.0, gap=0.0)
    cube_mass = 0.08  # kg, explicit in the Article 2 MJCF
    cube_inertia = wp.mat33(np.eye(3) * cube_mass * (2.0 * task.CUBE_HALF) ** 2 / 6.0)

    builder.add_shape_box(
        body=-1,
        hx=float(task.TABLE_HALF_SIZE[0]),
        hy=float(task.TABLE_HALF_SIZE[1]),
        hz=float(task.TABLE_HALF_SIZE[2]),
        xform=wp.transform(wp.vec3(*task.TABLE_CENTER), wp.quat_identity()),
        cfg=table_cfg,
        color=(0.32, 0.32, 0.32),
        label="table",
    )

    for label, pos, color in (
        ("red_cube", task.RED_CUBE_POS, (0.85, 0.05, 0.04)),
        ("blue_cube", task.BLUE_CUBE_POS, (0.05, 0.20, 0.90)),
    ):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), label=label,
            mass=cube_mass, inertia=cube_inertia, lock_inertia=True,
        )
        builder.add_shape_box(
            body=body,
            hx=task.CUBE_HALF,
            hy=task.CUBE_HALF,
            hz=task.CUBE_HALF,
            cfg=cube_cfg,
            color=color,
            label=label,
        )

    # Step 5: ground plane completes the single-world scene.
    builder.add_ground_plane()
    return builder


def build_newton_model(spec: RobotSpec | None = None) -> newton.Model:
    """Assemble the pick-and-place scene and pack it into device arrays."""
    return build_newton_builder(spec).finalize()


def body_index(model: newton.Model, suffix: str) -> int:
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


def set_joint_targets(control: newton.Control, ctrl: np.ndarray, target_indices: np.ndarray) -> None:
    """Update only robot position targets, using Model.joint_target_q_start."""
    ctrl = np.asarray(ctrl)
    target_indices = np.asarray(target_indices)
    if ctrl.shape != target_indices.shape or ctrl.ndim != 1:
        raise ValueError("Controller output must match the robot target index vector")
    targets = control.joint_target_q.numpy()
    targets[target_indices] = ctrl
    control.joint_target_q.assign(targets)
