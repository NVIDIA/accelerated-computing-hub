# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# Robot profiles for the pick-and-place task.
#
# The same scripts run the SO-101 (5-DoF + hinge gripper) and the Seeed reBot
# DevArm (6-DoF + coupled slide gripper). Switch with --robot so101|rebot.
# Layout numbers (table pose, cube spawn, seed configuration) live here so the
# IK / waypoint code in pick_place_common.py stays robot-agnostic.

from __future__ import annotations

from dataclasses import dataclass
import argparse

import numpy as np

# Pinned DeepMind menagerie commit used to validate the SO-101 task.
SO101_MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
SO101_MENAGERIE_REF = "feadf76d42f8a2162426f7d226a3b539556b3bf5"

# The reBot DevArm merged into upstream Menagerie (google-deepmind PR #300);
# pinned to the merge commit for reproducibility.
REBOT_MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
REBOT_MENAGERIE_REF = "da76818e269b82289eba39808e2fb91d679d6994"


@dataclass(frozen=True)
class RobotSpec:
    """Everything the pick-and-place task needs to know about one arm."""

    key: str
    display_name: str
    menagerie_url: str
    menagerie_ref: str
    folder: str
    robot_xml: str
    cache_dirname: str

    gripper_body: str
    left_jaw_geoms: tuple[str, ...]
    right_jaw_geoms: tuple[str, ...]
    left_jaw_body: str | None
    right_jaw_body: str | None
    gripper_actuator_names: tuple[str, ...]
    gripper_ctrl_indices: tuple[int, ...]
    free_arm_joints: tuple[int, ...]
    wrist_lock_index: int | None
    wrist_lock_value: float
    gripper_open: float
    gripper_closed: float
    seed_pose: tuple[float, ...]

    # None keeps the MJCF actuator_forcerange (reBot). A float boosts arm joints.
    arm_force_limit: float | None
    arm_effort_limit: float
    arm_target_ke: float
    arm_target_kd: float

    table_center_xyz: tuple[float, float, float]
    table_half_xyz: tuple[float, float, float]
    red_xy: tuple[float, float]
    blue_xy: tuple[float, float]
    cube_half: float

    camera_lookat: tuple[float, float, float]
    camera_distance: float
    nconmax: int
    njmax: int
    approach_height: float

    @property
    def home_ctrl(self) -> np.ndarray:
        return np.asarray(self.seed_pose, dtype=np.float64)

    @property
    def table_center(self) -> np.ndarray:
        return np.asarray(self.table_center_xyz, dtype=np.float64)

    @property
    def table_half_size(self) -> np.ndarray:
        return np.asarray(self.table_half_xyz, dtype=np.float64)

    @property
    def table_top_z(self) -> float:
        return self.table_center_xyz[2] + self.table_half_xyz[2]

    @property
    def cube_center_z(self) -> float:
        return self.table_top_z + self.cube_half

    @property
    def red_cube_pos(self) -> np.ndarray:
        return np.array([self.red_xy[0], self.red_xy[1], self.cube_center_z], dtype=np.float64)

    @property
    def blue_cube_pos(self) -> np.ndarray:
        return np.array([self.blue_xy[0], self.blue_xy[1], self.cube_center_z], dtype=np.float64)

    @property
    def stack_on_blue_z(self) -> float:
        return self.table_top_z + 3.0 * self.cube_half + 0.004


SO101 = RobotSpec(
    key="so101",
    display_name="SO-101",
    menagerie_url=SO101_MENAGERIE_URL,
    menagerie_ref=SO101_MENAGERIE_REF,
    folder="robotstudio_so101",
    robot_xml="so101.xml",
    cache_dirname="mujoco_menagerie",
    gripper_body="gripper",
    left_jaw_geoms=("fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2", "fixed_jaw_sph_tip3"),
    right_jaw_geoms=("moving_jaw_sph_tip1", "moving_jaw_sph_tip2", "moving_jaw_sph_tip3"),
    left_jaw_body=None,
    right_jaw_body=None,
    gripper_actuator_names=("gripper",),
    gripper_ctrl_indices=(5,),
    free_arm_joints=(0, 1, 2, 3),
    wrist_lock_index=4,
    wrist_lock_value=1.584,
    gripper_open=0.7,
    gripper_closed=0.30,
    seed_pose=(0.0, 0.0, 0.473, 1.177, 1.584, 0.7),
    arm_force_limit=30.0,
    arm_effort_limit=30.0,
    arm_target_ke=300.0,
    arm_target_kd=30.0,
    table_center_xyz=(0.35, -0.04, 0.012),
    table_half_xyz=(0.16, 0.26, 0.012),
    red_xy=(0.33, -0.13),
    blue_xy=(0.33, 0.06),
    cube_half=0.022,
    camera_lookat=(0.35, -0.04, 0.12),
    camera_distance=1.0,
    nconmax=128,
    njmax=300,
    approach_height=0.09,
)

# Seeed reBot DevArm: 6-DoF + 1:1 rack-and-pinion gripper (two slide joints).
# Fingers open along world Y at wrist roll = 0. Table is raised to the arm's
# natural workspace (~0.20 m) rather than forcing the arm down to a low table.
REBOT = RobotSpec(
    key="rebot",
    display_name="Seeed reBot DevArm",
    menagerie_url=REBOT_MENAGERIE_URL,
    menagerie_ref=REBOT_MENAGERIE_REF,
    folder="seeed_rebot_devarm",
    robot_xml="seeed_rebot_devarm.xml",
    cache_dirname="mujoco_menagerie_rebot",
    gripper_body="gripper_end",
    left_jaw_geoms=(),
    right_jaw_geoms=(),
    left_jaw_body="gripper_left",
    right_jaw_body="gripper_right",
    gripper_actuator_names=("joint_left", "joint_right"),
    gripper_ctrl_indices=(6, 7),
    free_arm_joints=(0, 1, 2, 3, 4),
    wrist_lock_index=5,
    wrist_lock_value=0.0,
    gripper_open=0.040,  # ~80 mm tip gap
    gripper_closed=0.018,  # ~36 mm, pinches the 44 mm cube
    # Raised keyframe: gripper well above the table so the bulky housing
    # does not start intersecting the table or the cubes.
    seed_pose=(0.0, -0.7, -1.1, 0.0, 0.0, 0.0, 0.040, 0.040),
    arm_force_limit=None,
    arm_effort_limit=36.0,
    arm_target_ke=400.0,
    arm_target_kd=40.0,
    table_center_xyz=(0.48, 0.0, 0.200),
    table_half_xyz=(0.16, 0.30, 0.012),
    red_xy=(0.48, -0.14),
    blue_xy=(0.48, 0.14),
    cube_half=0.022,
    camera_lookat=(0.48, 0.0, 0.24),
    camera_distance=1.4,
    nconmax=256,
    njmax=500,
    approach_height=0.12,
)

ROBOTS: dict[str, RobotSpec] = {
    "so101": SO101,
    "rebot": REBOT,
    "devarm": REBOT,
    "seeed": REBOT,
    "seeed_rebot_devarm": REBOT,
}

DEFAULT_ROBOT = "so101"


def get_robot(key: str | None = None) -> RobotSpec:
    """Return the profile for ``key`` (``so101`` or ``rebot``)."""
    name = (key or DEFAULT_ROBOT).strip().lower()
    if name not in ROBOTS:
        known = "so101, rebot"
        raise ValueError(f"Unknown robot {key!r}. Choose one of: {known}")
    return ROBOTS[name]


def add_robot_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robot",
        default=DEFAULT_ROBOT,
        choices=("so101", "rebot"),
        help="Manipulator to load (default: so101).",
    )
