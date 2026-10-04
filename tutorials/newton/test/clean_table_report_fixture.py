"""Synthetic protocol data for gate/exporter tests; never physics evidence."""


def make_gripper_report(robot="so101", device="cpu"):
    """Return a fresh internally consistent schema-2 report for isolated tests."""
    report = {
        "success": True, "finite": True, "failure": None,
        "schema_version": 2, "task": "gripper_pick_place_into_bin",
        "robot": robot, "device": device, "frames": 650, "simulation_seconds": 13.0,
        "phase": "done", "table_z": 0.1,
        "bin_lower": [0.0, -0.1, 0.0], "bin_upper": [0.2, 0.1, 0.08],
        "tool_withdrawn": True, "tool_clearance_m": 0.02,
        "max_coupling_input_force_norm": 0.2, "max_soft_contacts": 3,
        "mujoco_warp_overflow_flags": 0, "objects": {},
    }
    for index, name in enumerate(("red_cube", "blue_cube", "shirt", "cable")):
        start = 0.5 + index * 3.0
        report["objects"][name] = {
            "inside": True, "settled_frames": 50, "max_point_speed": 0.01,
            "bounds_min": [0.08, -0.02, 0.01], "bounds_max": [0.12, 0.02, 0.05],
            "initially_outside_bin": True,
            "grasped": True, "lifted": True, "carried": True,
            "over_bin_before_release": True, "release_commanded": True,
            "released": True, "dropped_before_release": False,
            "max_loaded_bilateral_frames": 60, "full_lift_frames": 25,
            "carry_frames": 10, "detached_settled_frames": 50,
            "grasp_time": start, "lift_time": start + 0.5,
            "carry_time": start + 1.0, "release_command_time": start + 1.2,
            "release_time": start + 1.4,
            "min_lift_clearance_m": 0.04, "min_jaw_normal_force_N": 0.2,
            "carry_bounds_min": [0.08, -0.02, 0.14], "carry_bounds_max": [0.12, 0.02, 0.18],
            "grasp_center": [0.1, -0.2, 0.12], "carry_start_center": [0.1, -0.2, 0.16],
            "carry_center": [0.1, 0.0, 0.16],
            "carry_distance_m": 0.2, "release_open_fraction": 0.9,
            "jaw_contacts": [0, 0],
        }
    return report
