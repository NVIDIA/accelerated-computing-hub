# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Optional, single-world measurements of the complete scripted workflow.

These rows include online inverse kinematics, control updates, and stepping.
Newton reconstructs the scene and position actuators, so these are workflow
cost comparisons, not the identical-model replay benchmark. They must not be
used to calculate the replay benchmark's CPU/GPU crossover.

``run_case`` is intended to run in an isolated worker process: the tutorial
helpers maintain robot-specific module globals. No renderer is constructed.
"""

from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import warp as wp


BACKENDS = ("mujoco", "newton_cpu", "newton_cuda")


@wp.kernel(enable_backward=False)
def _record_payload_frame(body_q: wp.array(dtype=wp.transform),
                          body_qd: wp.array(dtype=wp.spatial_vector),
                          sim_time: wp.array(dtype=float), red_id: int, blue_id: int,
                          frame: int, history: wp.array2d(dtype=float)):
    """Keep both cube histories on the simulation device until final download."""
    cube = wp.tid()
    body = red_id
    if cube == 1:
        body = blue_id
    if cube == 0:
        history[frame, 0] = sim_time[0]
    pose_start = 1 + cube * 7
    velocity_start = 15 + cube * 6
    position = wp.transform_get_translation(body_q[body])
    rotation = wp.transform_get_rotation(body_q[body])
    velocity = body_qd[body]
    for axis in range(3):
        history[frame, pose_start + axis] = position[axis]
        history[frame, pose_start + 4 + axis] = rotation[axis]
    history[frame, pose_start + 3] = rotation[3]  # Newton xyzw -> MuJoCo wxyz
    for axis in range(6):
        history[frame, velocity_start + axis] = velocity[axis]


def _validation_model(mujoco, spec):
    """Describe the common payload observation layout; never simulate this model."""
    half = spec.cube_half
    bodies = "".join(f'<body name="{name}"><freejoint/><geom type="box" size="{half} {half} {half}"/></body>'
                     for name in ("red_cube", "blue_cube"))
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.02"/><worldbody>' + bodies + '</worldbody></mujoco>'
    )


def validate_payload_history(history, spec, phases):
    """Use the replay's lift, airborne carry, table support, and settling gates.

    The workflow records at the 50 Hz controller rate instead of the replay's
    500 Hz physics rate. Cube poses use wxyz; linear/angular velocity magnitudes
    are invariant to the native/Newton angular-velocity frame convention.
    """
    import mujoco
    from benchmark_workload import validate_trajectory

    return validate_trajectory(history[None], _validation_model(mujoco, spec), spec, phases,
                               substeps=1, expected_steps=600)


def _configuration(config: dict[str, Any] | Any) -> dict[str, Any]:
    values = dict(config) if isinstance(config, dict) else vars(config).copy()
    for key, value in {
        "scope": "workflow", "backend": "mujoco", "robot": "so101",
        "device": "cuda:0", "frames": 600, "substeps": 10,
        "repeats": 3, "warmups": 1, "worlds": 1, "cpu_threads": 1,
        "nconmax": None, "njmax": None,
    }.items():
        values.setdefault(key, value)
    for key in ("frames", "substeps", "repeats", "warmups", "worlds", "cpu_threads"):
        if isinstance(values[key], bool) or not isinstance(values[key], Integral) or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
        values[key] = int(values[key])
    if values["scope"] != "workflow" or values["backend"] not in BACKENDS:
        raise ValueError("Workflow scope requires mujoco, newton_cpu, or newton_cuda")
    if values["robot"] not in ("so101", "rebot"):
        raise ValueError("Unknown robot")
    if values["frames"] != 600 or values["substeps"] != 10:
        raise ValueError("Workflow measurements require 600 frames and 10 substeps (12 simulated seconds)")
    if values["worlds"] != 1 or values["cpu_threads"] != 1:
        raise ValueError("Workflow measurements use one world and one CPU rollout thread")
    if values["repeats"] < 1 or values["warmups"] < 1:
        raise ValueError("At least one measured repeat and one full warmup episode are required")
    if values["nconmax"] is not None or values["njmax"] is not None:
        raise ValueError("Workflow capacities come from the unchanged tutorial; custom capacities are replay-only")
    if values["backend"] == "newton_cuda" and not str(values["device"]).startswith("cuda:"):
        raise ValueError("newton_cuda requires an explicit CUDA device such as cuda:0")
    return values


def _compiled_model_sha256(mujoco, np, model) -> str:
    """Fingerprint the actual compiled model, including in-memory overrides."""
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _validate_native(mujoco, np, model, data, controller, red, blue) -> dict[str, Any]:
    """Apply the completed Newton tutorial's final task gates to native MuJoCo."""
    if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc, data.xpos)):
        raise ValueError("Nonfinite MuJoCo state")
    warnings = {mujoco.mjtWarning(index).name: int(warning.number)
                for index, warning in enumerate(data.warning) if warning.number}
    if warnings:
        raise ValueError(f"MuJoCo warnings: {warnings}")
    if not controller.done:
        raise ValueError("Pick-and-place sequence is incomplete")
    return _stack_metrics(np, red, blue)


def _stack_metrics(np, red, blue) -> dict[str, Any]:
    if not np.isfinite(red).all() or not np.isfinite(blue).all():
        raise ValueError("Nonfinite cube position")
    xy_error = float(np.linalg.norm(red[:2] - blue[:2]))
    height = float(red[2] - blue[2])
    if xy_error > 0.015 or not 0.035 <= height <= 0.055:
        raise ValueError(f"Red cube did not end stacked on blue: xy={xy_error:.6g}, dz={height:.6g}")
    return {"red_xyz": red.tolist(), "blue_xyz": blue.tolist(),
            "xy_error_m": xy_error, "height_difference_m": height}


def run_case(config: dict[str, Any] | Any) -> dict[str, Any]:
    """Return one JSON-safe workflow row, with setup and validation untimed.

    Timed episodes contain all 600 online controller updates and 6,000 physics
    steps, including Newton's target transfers and graph launches. Every
    episode gets a fresh model, data, controller, and solver outside that timer.
    Full-episode warmups precede measured repeats. CUDA is synchronized at
    timer boundaries. Final host-state collection and strict validation each
    have separate timers; samples expose their sum as ``end_to_end_seconds``.
    """
    started = time.perf_counter()
    try:
        config = _configuration(config)
    except (TypeError, ValueError) as error:
        supplied = dict(config) if isinstance(config, dict) else vars(config).copy()
        return {"status": "failed", "scope": "workflow", "backend": supplied.get("backend", "mujoco"),
                "robot": supplied.get("robot", "so101"), "worlds": supplied.get("worlds", 1),
                "samples": [], "warmup_samples": [], "workload": {}, "config": supplied,
                "timings": {"case_seconds": time.perf_counter() - started},
                "reason": f"{type(error).__name__}: {error}",
                "diagnostics": {"error_type": type(error).__name__, "error": str(error)}}
    row: dict[str, Any] = {
        "scope": "workflow", "backend": config["backend"], "robot": config["robot"],
        "worlds": 1, "status": "failed", "config": config, "samples": [], "warmup_samples": [],
        "timings": {"preparation_seconds": 0.0, "backend_setup_seconds": 0.0,
                    "warmup_seconds": 0.0, "episode_setup_seconds": []},
        "workload": {
            "task": "scripted_pick_and_place_stack", "frames": 600,
            "physics_steps": 6000, "simulated_seconds": 12.0,
            "control_hz": 50, "physics_dt_seconds": 0.002,
            "controller": "online waypoint inverse kinematics, one CPU rollout thread",
            "model_equivalence": "workflow-only; Newton reconstructs the scene and actuation",
            "model_sha256": None, "controls_sha256": None,
            "control_policy": "computed online; no prerecorded controls",
            "rendering": False, "contact_detection": "MuJoCo for all backends",
            "output_policy": "600 payload pose/velocity observations at 50 Hz, plus final body state; final host transfer separately timed",
            "output_layout_note": "Native float64 body arrays include the world body; Newton float32 arrays follow Newton body indexing. Native output refreshes final FK.",
            "timed_scope": "online IK, control updates/transfers, physics stepping, payload history capture in backend memory, CUDA synchronization",
            "observation_policy": "CPU writes payload history to RAM; CUDA launches one recorder kernel per control frame and downloads once after the episode. Observation overhead is included in simulation_seconds.",
            "compilation_policy": "Construction, graph capture, and their JIT work are setup; any remaining lazy compilation occurs in full-episode warmups. The on-disk Warp cache may be warm.",
        },
        "diagnostics": {
            "validation": "controller complete, finite history, lift/carry, table support, last-second settling, final stack; capacity/warning guards",
            "model_differences": [
                "Native MuJoCo loads the task MJCF; Newton imports the robot and reconstructs table, cubes, and ground.",
                "Newton uses coordinate position targets and reconstructed gains; original MuJoCo uses its MJCF actuators.",
                "Newton explicitly selects elliptic contacts, implicitfast, 100 solver iterations and 50 line-search iterations.",
                "These workflow rows are excluded from identical-model replay speedups and crossover calculations.",
            ],
        },
    }
    try:
        import mujoco
        import numpy as np

        part = Path(__file__).resolve().parent
        sys.path.insert(0, str(part))
        import pick_place_common as task
        from robots import get_robot

        spec = get_robot(config["robot"])
        wp = None
        Example = None
        ViewerNull = None
        if config["backend"] != "mujoco":
            import warp as wp
            wp.init()
            requested = config["device"] if config["backend"] == "newton_cuda" else "cpu"
            if config["backend"] == "newton_cuda" and not wp.is_cuda_available():
                row.update(status="skipped", reason="No CUDA device is available; no CPU substitution was made")
                row["timings"]["preparation_seconds"] = time.perf_counter() - started
                return row
            wp.set_device(requested)
            if config["backend"] == "newton_cuda" and not wp.get_device(requested).is_cuda:
                raise ValueError("Requested workflow device is not CUDA")
            sys.path.insert(0, str(part / "solutions"))
            from so101_newton_solution import Example
            from newton.viewer import ViewerNull

        row["device"] = config["device"] if config["backend"] == "newton_cuda" else "cpu"
        row["device_info"] = {"device": row["device"], "cuda": config["backend"] == "newton_cuda",
                              "cpu_rollout_threads": 1}
        if wp is not None:
            device = wp.get_device(row["device"])
            row["device_info"].update(name=str(device.name), arch=str(device.arch))
        row["workload"]["asset_ref"] = spec.menagerie_ref
        source_names = ("benchmark_workflows.py", "pick_place_common.py", "robots.py", "utils.py")
        source_names += ("benchmark_workload.py",)
        if Example is not None:
            source_names += ("solutions/so101_newton_solution.py", "solutions/newton_scene_solution.py")
        row["workload"]["source_sha256"] = {
            name: hashlib.sha256((part / name).read_bytes()).hexdigest() for name in source_names
        }
        row["timings"]["preparation_seconds"] = time.perf_counter() - started

        def synchronize():
            if config["backend"] == "newton_cuda":
                wp.synchronize_device(config["device"])

        for episode in range(config["warmups"] + config["repeats"]):
            warmup = episode < config["warmups"]
            setup_started = time.perf_counter()
            example = None
            if config["backend"] == "mujoco":
                scene = task.resolve_pick_place_scene(spec=spec)
                model = task.load_pick_place_model(scene, spec)
                model.opt.timestep = 0.002
                data = mujoco.MjData(model)
                task.apply_arm_ctrl(model, data, spec.home_ctrl)
                task.reset_cubes(model, data, spec)
                controller = task.PickPlaceController(spec=spec)
                red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
                blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")
                cube_addresses = [(int(model.jnt_qposadr[model.body_jntadr[index]]),
                                   int(model.jnt_dofadr[model.body_jntadr[index]]))
                                  for index in (red_id, blue_id)]
            else:
                example = Example(ViewerNull(num_frames=600),
                                  SimpleNamespace(robot=spec.key, use_mujoco_contacts=True))
                if example.use_newton_contacts:
                    raise ValueError("Workflow comparisons require MuJoCo contact detection on both devices")
                model = example.solver.mj_model
                model.opt.timestep = 0.002
                controller = example.controller
                red_id, blue_id = example.red_body, example.blue_body
            phases = []
            payload_history = np.empty((600, 27), dtype=np.float64)
            device_history = wp.zeros((600, 27), dtype=float, device=config["device"]) \
                if config["backend"] == "newton_cuda" else None
            synchronize()
            model_hash = _compiled_model_sha256(mujoco, np, model)
            if row["workload"]["model_sha256"] not in (None, model_hash):
                raise ValueError("Compiled model changed between fresh episodes")
            row["workload"]["model_sha256"] = model_hash
            row["workload"]["model_mjb_sha256"] = model_hash
            row["workload"]["model_dimensions"] = {
                "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
                "nbody": int(model.nbody), "ngeom": int(model.ngeom),
            }
            row["workload"]["solver_options"] = {
                "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
                "cone": int(model.opt.cone), "iterations": int(model.opt.iterations),
                "ls_iterations": int(model.opt.ls_iterations), "timestep": float(model.opt.timestep),
                "tolerance": float(model.opt.tolerance), "ls_tolerance": float(model.opt.ls_tolerance),
                "impratio": float(model.opt.impratio), "gravity": model.opt.gravity.tolist(),
            }
            row["diagnostics"]["actual_capacities"] = {
                "nconmax": int(example.solver.mjw_data.naconmax),
                "njmax": int(example.solver.mjw_data.njmax),
            } if config["backend"] == "newton_cuda" else None
            setup_seconds = time.perf_counter() - setup_started
            row["timings"]["backend_setup_seconds"] += setup_seconds
            row["timings"]["episode_setup_seconds"].append({"warmup": warmup, "seconds": setup_seconds})

            synchronize()
            simulation_started = time.perf_counter()
            for frame in range(600):
                phases.append(controller.phase_name())
                if example is None:
                    control = controller.step(model, data, 0.02)
                    for _ in range(10):
                        data.ctrl[:model.nu] = control
                        mujoco.mj_step(model, data)
                    payload_history[frame, 0] = data.time
                    for cube, (qpos, qvel) in enumerate(cube_addresses):
                        payload_history[frame, 1 + cube * 7:8 + cube * 7] = data.qpos[qpos:qpos + 7]
                        payload_history[frame, 15 + cube * 6:21 + cube * 6] = data.qvel[qvel:qvel + 6]
                else:
                    example.step()
                    if device_history is not None:
                        wp.launch(_record_payload_frame, dim=2, inputs=[
                            example.state_0.body_q, example.state_0.body_qd, example.solver.mjw_data.time,
                            red_id, blue_id, frame, device_history,
                        ], device=example.model.device)
                    else:
                        poses = example.state_0.body_q.numpy()
                        velocities = example.state_0.body_qd.numpy()
                        payload_history[frame, 0] = example.solver.mj_data.time
                        for cube, body in enumerate((red_id, blue_id)):
                            payload_history[frame, 1 + cube * 7:8 + cube * 7] = poses[body, [0, 1, 2, 6, 3, 4, 5]]
                            payload_history[frame, 15 + cube * 6:21 + cube * 6] = velocities[body]
            synchronize()
            simulation_seconds = time.perf_counter() - simulation_started

            output_started = time.perf_counter()
            if example is None:
                # mj_step leaves position-dependent fields at the pre-integration
                # configuration; refresh them before exporting the final state.
                mujoco.mj_forward(model, data)
                body_pose = np.concatenate((data.xpos, data.xquat), axis=1).copy()
                body_velocity = data.cvel.copy()
            else:
                body_pose = example.state_0.body_q.numpy().copy()
                body_velocity = example.state_0.body_qd.numpy().copy()
                if device_history is not None:
                    payload_history = device_history.numpy()
            synchronize()
            output_seconds = time.perf_counter() - output_started
            row["workload"]["payload_history_dtype"] = str(payload_history.dtype)
            row["workload"]["payload_observations"] = 600
            sample = {
                "repeat": episode if warmup else episode - config["warmups"],
                "simulation_seconds": simulation_seconds,
                "output_transfer_seconds": output_seconds,
                "validation_seconds": 0.0, "task_success_count": 0,
                "task_total_count": 1, "passed": False,
                "final_host_output_bytes": int(body_pose.nbytes + body_velocity.nbytes),
                "payload_history_bytes": int(payload_history.nbytes),
            }
            validation_started = time.perf_counter()
            try:
                red, blue = body_pose[red_id, :3], body_pose[blue_id, :3]
                if not np.isfinite(body_velocity).all():
                    raise ValueError("Nonfinite final body velocity")
                if example is None:
                    sample["task_metrics"] = _validate_native(mujoco, np, model, data, controller, red, blue)
                else:
                    # Keep the actual solution's capacity, finite-state and stack
                    # checks. Sticky CUDA overflow flags retain substep failures.
                    example.test_final()
                    sample["task_metrics"] = _stack_metrics(np, red, blue)
                    if config["backend"] == "newton_cpu":
                        warnings = [int(warning.number) for warning in example.solver.mj_data.warning]
                        if any(warnings):
                            raise ValueError(f"Native Newton/MuJoCo warnings: {warnings}")
                    else:
                        import mujoco_warp as mjw
                        flags = example.solver.mjw_data.overflow.numpy()
                        iterations = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
                        sample["overflow_flags"] = flags.tolist()
                        sample["capacity_flags"] = (flags & ~iterations).tolist()
                        sample["iteration_warning_flags"] = (flags & iterations).tolist()
                sample["validation"] = validate_payload_history(payload_history, spec, np.asarray(phases))
                if not sample["validation"]["passed"]:
                    raise ValueError("Payload history failed lift/carry, table support, or settling validation")
                sample.update(passed=True, task_success_count=1)
            except Exception as error:
                sample["error"] = f"{type(error).__name__}: {error}"
            sample["validation_seconds"] = time.perf_counter() - validation_started
            sample.setdefault("validation", {
                "passed": sample["passed"], "task_success_count": sample["task_success_count"],
                "task_total_count": 1, "worlds": [sample.get("task_metrics", {})],
            })
            sample["end_to_end_seconds"] = sum(sample[key] for key in
                ("simulation_seconds", "output_transfer_seconds", "validation_seconds"))
            if warmup:
                row["warmup_samples"].append(sample)
                row["timings"]["warmup_seconds"] += sample["end_to_end_seconds"]
            else:
                row["samples"].append(sample)
            if not sample["passed"]:
                row["reason"] = sample["error"]
                row["diagnostics"]["error"] = sample["error"]
                return row
        row["status"] = "passed"
    except Exception as error:
        row["reason"] = f"{type(error).__name__}: {error}"
        row["diagnostics"].update(error_type=type(error).__name__, error=str(error))
    finally:
        row["timings"]["case_seconds"] = time.perf_counter() - started
    return row
