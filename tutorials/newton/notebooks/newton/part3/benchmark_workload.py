# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Complete, checked pick-and-place replay on native MuJoCo or MuJoCo Warp.

``run_case`` measures the same 12-second, open-loop task on either backend.
Both produce FULLPHYSICS states after every physics step in backend memory;
GPU download and task validation are timed separately. This is not a policy,
rendering, Newton/VBD coupling, or end-to-end training benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
from mujoco import rollout
import numpy as np
import warp as wp

import pick_place_common as task
from robots import get_robot


@wp.kernel(enable_backward=False)
def _load_control(tape: wp.array2d(dtype=float), cursor: wp.array(dtype=int),
                  substeps: int, ctrl: wp.array2d(dtype=float)):
    world, actuator = wp.tid()
    ctrl[world, actuator] = tape[cursor[0] // substeps, actuator]


@wp.kernel(enable_backward=False)
def _record_state(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
                  act: wp.array2d(dtype=float), sim_time: wp.array(dtype=float),
                  cursor: wp.array(dtype=int), nq: int, nv: int,
                  trajectory: wp.array3d(dtype=float)):
    world, field = wp.tid()
    value = float(0.0)
    if field == 0:
        value = sim_time[world]
    elif field <= nq:
        value = qpos[world, field - 1]
    elif field <= nq + nv:
        value = qvel[world, field - 1 - nq]
    else:
        value = act[world, field - 1 - nq - nv]
    trajectory[world, cursor[0], field] = value


@wp.kernel(enable_backward=False)
def _record_capacity(nacon: wp.array(dtype=int), nefc: wp.array(dtype=int),
                     contact_peak: wp.array(dtype=int), constraint_peak: wp.array(dtype=int)):
    world = wp.tid()
    if world == 0:
        contact_peak[0] = wp.max(contact_peak[0], nacon[0])
    constraint_peak[world] = wp.max(constraint_peak[world], nefc[world])


@wp.kernel(enable_backward=False)
def _advance_cursor(cursor: wp.array(dtype=int)):
    cursor[0] = cursor[0] + 1


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _positive_int(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def _configuration(config: dict) -> dict:
    backend = config.get("backend", "mujoco")
    if backend not in ("mujoco", "mjwarp"):
        raise ValueError("backend must be mujoco or mjwarp")
    robot = config.get("robot", "so101")
    if robot not in ("so101", "rebot"):
        raise ValueError("robot must be so101 or rebot")
    result = dict(config, backend=backend, robot=robot)
    for key, default in (("worlds", 1), ("cpu_threads", 1), ("repeats", 3),
                         ("warmups", 1), ("frames", 600), ("substeps", 10)):
        result[key] = _positive_int(config, key, default)
    if result["frames"] != 600 or result["substeps"] != 10:
        raise ValueError("The benchmark requires the complete 600-frame task with 10 substeps")
    for key in ("nconmax", "njmax"):
        if config.get(key) is not None:
            result[key] = _positive_int(config, key, 1)
    result["device"] = str(config.get("device", "cuda:0"))
    if backend == "mjwarp" and not result["device"].startswith("cuda:"):
        raise ValueError("MJWarp benchmark requires an explicit CUDA device; no CPU fallback")
    limit = float(config.get("max_trajectory_mib", 1024))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("max_trajectory_mib must be finite and positive")
    result["max_trajectory_mib"] = limit
    return result


def _prepare(config: dict):
    spec = get_robot(config["robot"])
    asset_path = config.get("menagerie_path")
    scene = task.resolve_pick_place_scene(
        Path(asset_path) if asset_path else None, explicit=bool(asset_path), spec=spec,
    )
    model = task.load_pick_place_model(scene, spec)
    model.opt.timestep = 0.002
    # Shared quality configuration. SO-101's source XML uses 10/20, whereas
    # this comparison requires a settled stack on both arithmetic backends.
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    model.opt.impratio = 100
    initial = mujoco.MjData(model)
    task.apply_arm_ctrl(model, initial, spec.home_ctrl)
    task.reset_cubes(model, initial, spec)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    if nstate != 1 + model.nq + model.nv + model.na or model.nsensordata:
        raise ValueError("This replay supports the tutorial models without history, plugin state or sensors")
    initial_state = np.empty(nstate, dtype=np.float64)
    mujoco.mj_getState(model, initial, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    controller = task.PickPlaceController(spec=spec)
    commands, phases = [], []
    for _ in range(config["frames"]):
        phases.append(controller.phase_name())
        commands.append(controller.step(model, initial, 0.02).copy())
    if not controller.done:
        raise ValueError("The complete command tape did not finish the controller sequence")
    # Identical command values: native MuJoCo uses the float64 representation of
    # the float32 command values applied by MJWarp, not a different IK tape.
    tape = np.asarray(commands, dtype=np.float32)
    if not np.isfinite(tape).all():
        raise ValueError("Nonfinite controller tape")
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    source_files = ("benchmark_workload.py", "pick_place_common.py", "robots.py", "utils.py")
    source_hashes = {name: _sha((Path(__file__).parent / name).read_bytes()) for name in source_files}
    model_options = {
        "timestep": float(model.opt.timestep), "solver": int(model.opt.solver),
        "integrator": int(model.opt.integrator), "cone": int(model.opt.cone),
        "iterations": int(model.opt.iterations), "ls_iterations": int(model.opt.ls_iterations),
        "tolerance": float(model.opt.tolerance), "ls_tolerance": float(model.opt.ls_tolerance),
        "impratio": float(model.opt.impratio), "gravity": model.opt.gravity.tolist(),
        "disableflags": int(model.opt.disableflags), "enableflags": int(model.opt.enableflags),
    }
    workload = {
        "task": "pick_red_cube_and_stack_on_blue", "frames": config["frames"],
        "substeps": config["substeps"], "physics_steps": config["frames"] * config["substeps"],
        "dt": float(model.opt.timestep), "control_dt": 0.02, "simulated_seconds": 12.0,
        "initial_states": "identical replicated task; no domain randomization",
        "controller": "precomputed open-loop waypoint/IK tape",
        "control_dtype": "float32 values, cast to float64 for native MuJoCo",
        "output": "FULLPHYSICS after each physics step; no sensors",
        "model_dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "na", "nbody", "ngeom")},
        "model_options": model_options, "model_mjb_sha256": _sha(binary.tobytes()),
        "solver_configuration": "100 solver / 50 line-search iterations and impratio=100 on BOTH backends; overrides source XML settings",
        "acceptance": {
            "whole_cube_lift_m": 0.01, "minimum_lift_frames": 10,
            "uninterrupted_airborne_carry_m": 0.05, "minimum_carry_frames": 10,
            "stack_xy_error_m": 0.015, "stack_height_difference_m": [0.035, 0.055],
            "settling_seconds": 1.0, "rms_cube_point_speed_bound_m_s": 0.05,
            "instantaneous_point_speed_maximum": "reported as a diagnostic, not a rejection threshold",
            "max_position_diameter_m": 0.005, "max_orientation_diameter_deg": 3.0,
            "blue_table_clearance_tolerance_m": 0.005,
        },
        "scene_xml_sha256": _sha(scene.read_bytes()), "robot_xml_sha256": _sha((scene.parent / spec.robot_xml).read_bytes()),
        "menagerie_ref": spec.menagerie_ref, "explicit_asset_path": bool(asset_path),
        "control_tape_sha256": _sha(tape.astype("<f4", copy=False).tobytes()),
        "initial_state_sha256": _sha(initial_state.astype("<f8", copy=False).tobytes()),
        "source_sha256": source_hashes,
        "simulation_timer": "state reset, command replay, physics and FULLPHYSICS recording; CUDA synchronized",
    }
    comparison = {key: workload[key] for key in (
        "task", "frames", "substeps", "physics_steps", "dt", "control_dt",
        "model_mjb_sha256", "control_tape_sha256", "initial_state_sha256", "source_sha256",
    )}
    workload["comparison_signature"] = _sha(json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode())
    return model, initial, initial_state, tape, np.asarray(phases), spec, workload


def _cube_indices(model, name: str) -> tuple[int, int]:
    body = int(model.body(name).id)
    joint = int(model.body_jntadr[body])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{name} must have a free joint")
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def _cube_clearance(pose: np.ndarray, half: float, table_top: float) -> np.ndarray:
    """Lowest corner of an oriented cube above the tabletop (quaternion wxyz)."""
    quat = pose[..., 3:7]
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norms, 1e-30)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    extent_z = half * (np.abs(2 * (x * z - w * y)) +
                       np.abs(2 * (y * z + w * x)) + np.abs(1 - 2 * (x * x + y * y)))
    return pose[..., 2] - extent_z - table_top


def _longest_true_run(values: np.ndarray) -> int:
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.max(ends - starts, initial=0))


def _continuous_carry_distance(positions: np.ndarray, airborne: np.ndarray, min_frames: int = 10) -> float:
    """Distance from the start of one uninterrupted airborne transport span."""
    edges = np.diff(np.r_[False, airborne, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return max((float(np.max(np.linalg.norm(positions[start:end] - positions[start], axis=1)))
                for start, end in zip(starts, ends) if end - start >= min_frames), default=0.0)


def _pose_diameter(pose: np.ndarray) -> tuple[float, float]:
    """Largest pairwise translation and rotation difference; no smoothing."""
    pose = np.asarray(pose, dtype=np.float64)
    position = float(np.max(np.linalg.norm(pose[:, None, :3] - pose[None, :, :3], axis=-1)))
    quat = pose[:, 3:7]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-30)
    angle = float(np.degrees(2 * np.arccos(np.clip(np.min(np.abs(quat @ quat.T)), 0, 1))))
    return position, angle


def validate_trajectory(states: np.ndarray, model, spec, phases: np.ndarray, *,
                        substeps: int = 10, expected_steps: int = 6000) -> dict:
    """Check every world; diagnostic pose fixtures are not physical rollouts."""
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    if states.ndim != 3 or states.shape[1:] != (expected_steps, nstate):
        raise ValueError(f"Unexpected trajectory shape {states.shape}")
    rq, rv = _cube_indices(model, "red_cube")
    bq, bv = _cube_indices(model, "blue_cube")
    world_results = []
    for index, state in enumerate(states):
        finite = bool(np.isfinite(state).all())
        times = state[:, 0]
        time_ok = bool(finite and np.all(np.diff(times) > 0) and
                       np.allclose(np.diff(times), model.opt.timestep, rtol=0, atol=1e-5) and
                       abs(float(times[0]) - model.opt.timestep) < 1e-5 and
                       abs(float(times[-1]) - expected_steps * model.opt.timestep) < 0.002)
        if not finite:
            world_results.append({"world": index, "passed": False, "finite": False,
                                  "time_progression": False, "reason": "nonfinite state"})
            continue
        frames = state[substeps - 1::substeps]
        qpos = frames[:, 1:1 + model.nq]
        qvel = frames[:, 1 + model.nq:1 + model.nq + model.nv]
        red, blue = qpos[:, rq:rq + 7], qpos[:, bq:bq + 7]
        clearance = _cube_clearance(red, spec.cube_half, spec.table_top_z)
        blue_clearance = _cube_clearance(blue, spec.cube_half, spec.table_top_z)
        # The support cube must remain on the tabletop. A stack that fell to
        # the floor is not a successful placement, even if relative poses match.
        blue_inside_table = np.all(
            np.abs(blue[:, :2] - spec.table_center[:2]) <= spec.table_half_size[:2] - spec.cube_half,
            axis=1,
        )
        blue_supported = (np.abs(blue_clearance) <= 0.005) & blue_inside_table
        airborne = clearance > 0.01
        lift_frames = _longest_true_run(airborne & np.isin(phases, ["lift_red", "above_blue"]))
        carry_distance = _continuous_carry_distance(red[:, :2], (phases == "above_blue") & airborne)
        xy_error = np.linalg.norm(red[:, :2] - blue[:, :2], axis=1)
        dz = red[:, 2] - blue[:, 2]
        stacked = (xy_error <= 0.015) & (dz >= 0.035) & (dz <= 0.055)
        red_speed = np.linalg.norm(qvel[:, rv:rv + 3], axis=1)
        blue_speed = np.linalg.norm(qvel[:, bv:bv + 3], axis=1)
        red_spin = np.linalg.norm(qvel[:, rv + 3:rv + 6], axis=1)
        blue_spin = np.linalg.norm(qvel[:, bv + 3:bv + 6], axis=1)
        # Every point is at most sqrt(3)*half from the cube center. The rigid
        # velocity bound |v| + radius*|omega| gives a common physical quantity
        # for translation and rotation. Its one-second RMS measures sustained
        # resting motion; report isolated peaks separately. Raw pose diameters
        # still reject drift, wobble and lost placement without smoothing.
        radius = np.sqrt(3) * spec.cube_half
        red_point_speed = (red_speed + radius * red_spin)[-50:]
        blue_point_speed = (blue_speed + radius * blue_spin)[-50:]
        point_speed = float(max(np.max(red_point_speed), np.max(blue_point_speed)))
        point_speed_rms = float(max(np.sqrt(np.mean(red_point_speed**2)),
                                    np.sqrt(np.mean(blue_point_speed**2))))
        red_drift, red_angle = _pose_diameter(red[-50:])
        blue_drift, blue_angle = _pose_diameter(blue[-50:])
        position_diameter, orientation_diameter = max(red_drift, blue_drift), max(red_angle, blue_angle)
        settled = bool(np.all(stacked[-50:] & blue_supported[-50:]) and point_speed_rms <= 0.05 and
                       position_diameter <= 0.005 and orientation_diameter <= 3.0)
        passed = bool(time_ok and lift_frames >= 10 and carry_distance >= 0.05 and settled)
        world_results.append({
            "world": index, "passed": passed, "finite": finite, "time_progression": time_ok,
            "lift_frames": lift_frames, "airborne_carry_m": carry_distance,
            "max_cube_clearance_m": float(np.max(clearance)), "final_stack": bool(stacked[-1]),
            "blue_supported_on_table_last_1s": bool(np.all(blue_supported[-50:])),
            "settled_last_1s": settled, "final_xy_error_m": float(xy_error[-1]),
            "final_height_difference_m": float(dz[-1]), "final_time_s": float(times[-1]),
            "settling_max_linear_speed_m_s": float(max(np.max(red_speed[-50:]), np.max(blue_speed[-50:]))),
            "settling_max_angular_speed_rad_s": float(max(np.max(red_spin[-50:]), np.max(blue_spin[-50:]))),
            "settling_max_point_speed_bound_m_s": point_speed,
            "settling_rms_point_speed_bound_m_s": point_speed_rms,
            "settling_position_diameter_m": position_diameter,
            "settling_orientation_diameter_deg": orientation_diameter,
        })
    return {"passed": all(item["passed"] for item in world_results),
            "task_success_count": sum(item["passed"] for item in world_results),
            "task_total_count": len(world_results), "worlds": world_results}


class _CPUReplay:
    def __init__(self, config, model, initial, initial_state, tape):
        self.model = model
        worlds = config["worlds"]
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        self.threads = min(config["cpu_threads"], worlds, affinity)
        self.pool = rollout.Rollout(nthread=self.threads if self.threads > 1 else 0)
        self.data = [mujoco.MjData(model) for _ in range(self.threads)]
        self.models = [model] * worlds
        self.initial = np.tile(initial_state, (worlds, 1))
        self.warmstart = np.zeros((worlds, model.nv), dtype=np.float64)
        step_tape = np.repeat(tape.astype(np.float64), config["substeps"], axis=0)
        self.controls = np.tile(step_tape[None], (worlds, 1, 1))
        self.steps = len(step_tape)
        self.output = np.empty((worlds, self.steps, len(initial_state)), dtype=np.float64)
        self.sensors = np.empty((worlds, self.steps, 0), dtype=np.float64)
        self.info = {"kind": "cpu", "cpu_threads_requested": config["cpu_threads"],
                     "cpu_threads": self.threads, "output_dtype": "float64",
                     "trajectory_bytes": self.output.nbytes, "control_bytes": self.controls.nbytes}

    def episode(self):
        # rollout is stateless: it resets each world from initial and warmstart.
        start = time.perf_counter()
        self.pool.rollout(self.models, self.data, self.initial, self.controls, skip_checks=True,
                          nstep=self.steps, initial_warmstart=self.warmstart,
                          state=self.output, sensordata=self.sensors)
        elapsed = time.perf_counter() - start
        return self.output, elapsed, 0.0, {"capacity_passed": True, "storage": "native dynamic MuJoCo"}

    def close(self):
        self.pool.close()


def _capacity_diagnostics(flags, contacts, constraints, *, iteration_mask, naconmax, njmax, cursor, steps):
    capacity_flags = flags & ~iteration_mask
    return {
        "capacity_passed": bool(not np.any(capacity_flags) and contacts <= naconmax and
                                np.all(constraints <= njmax) and cursor == steps),
        "overflow_flags_per_world": flags.tolist(), "capacity_flags_per_world": capacity_flags.tolist(),
        "iteration_warning_flags_per_world": (flags & iteration_mask).tolist(),
        "peak_contacts_all_worlds": int(contacts), "peak_constraints_per_world": constraints.tolist(),
        "recorded_physics_steps": int(cursor),
    }


class _GPUReplay:
    def __init__(self, config, model, initial, initial_state, tape):
        import mujoco_warp as mjw

        self.mjw = mjw
        self.device = wp.get_device(config["device"])
        if not self.device.is_cuda:
            raise ValueError("MJWarp benchmark requires CUDA; no CPU fallback")
        self.config, self.native_model = config, model
        self.worlds, self.nstate = config["worlds"], len(initial_state)
        self.steps = config["frames"] * config["substeps"]
        self.nconmax = config.get("nconmax") or get_robot(config["robot"]).nconmax
        self.njmax = config.get("njmax") or get_robot(config["robot"]).njmax
        with wp.ScopedDevice(self.device):
            free_before = self.device.free_memory
            self.model = mjw.put_model(model)
            # Avoid printf costs in measurements; sticky diagnostic bits and
            # raw high-water marks remain enabled and are checked every run.
            self.model.opt.warn_overflow = False
            self.data = mjw.make_data(model, nworld=self.worlds, nconmax=self.nconmax, njmax=self.njmax)
            self.tape = wp.array(tape, dtype=wp.float32)
            self.initial_qpos = wp.array(np.tile(initial.qpos, (self.worlds, 1)), dtype=wp.float32)
            self.initial_qvel = wp.array(np.tile(initial.qvel, (self.worlds, 1)), dtype=wp.float32)
            self.initial_ctrl = wp.array(np.tile(initial.ctrl, (self.worlds, 1)), dtype=wp.float32)
            self.initial_act = wp.array(np.tile(initial.act, (self.worlds, 1)), dtype=wp.float32)
            self.cursor = wp.zeros(1, dtype=wp.int32)
            self.contact_peak = wp.zeros(1, dtype=wp.int32)
            self.constraint_peak = wp.zeros(self.worlds, dtype=wp.int32)
            self.output = wp.empty((self.worlds, self.steps, self.nstate), dtype=wp.float32)
            self.reset()
            # Compile all kernels before graph capture; this is setup, not a
            # measured rollout. Every later episode starts with another reset.
            self.frame()
            wp.synchronize_device(self.device)
            self.reset()
            with wp.ScopedCapture(device=self.device) as capture:
                self.frame()
            self.graph = capture.graph
            wp.synchronize_device(self.device)
            self.info = {
                "kind": "cuda", "name": self.device.name, "alias": self.device.alias,
                "architecture": int(self.device.arch), "output_dtype": "float32",
                "total_memory_bytes": int(self.device.total_memory),
                "free_memory_before_bytes": int(free_before),
                "free_memory_after_setup_bytes": int(self.device.free_memory),
                "trajectory_bytes": self.worlds * self.steps * self.nstate * 4,
                "control_bytes": int(tape.nbytes), "nconmax": self.nconmax,
                "naconmax": int(self.data.naconmax), "njmax": self.njmax,
            }

    def reset(self):
        self.mjw.reset_data(self.model, self.data)
        wp.copy(self.data.qpos, self.initial_qpos)
        wp.copy(self.data.qvel, self.initial_qvel)
        wp.copy(self.data.ctrl, self.initial_ctrl)
        if self.native_model.na:
            wp.copy(self.data.act, self.initial_act)
        self.cursor.zero_()
        self.contact_peak.zero_()
        self.constraint_peak.zero_()
        self.mjw.forward(self.model, self.data)
        self.data.qacc_warmstart.zero_()

    def frame(self):
        data, native = self.data, self.native_model
        wp.launch(_load_control, (self.worlds, native.nu),
                  inputs=[self.tape, self.cursor, self.config["substeps"], data.ctrl])
        for _ in range(self.config["substeps"]):
            self.mjw.step(self.model, data)
            wp.launch(_record_capacity, self.worlds,
                      inputs=[data.nacon, data.nefc, self.contact_peak, self.constraint_peak])
            wp.launch(_record_state, (self.worlds, self.nstate), inputs=[
                data.qpos, data.qvel, data.act, data.time, self.cursor, native.nq, native.nv, self.output,
            ])
            wp.launch(_advance_cursor, 1, inputs=[self.cursor])

    def episode(self):
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            start = time.perf_counter()
            # Native rollout includes its state reset; include the CUDA reset
            # here too. Compilation/allocation remains in backend setup.
            self.reset()
            for _ in range(self.config["frames"]):
                wp.capture_launch(self.graph)
            wp.synchronize_device(self.device)
            elapsed = time.perf_counter() - start
            start = time.perf_counter()
            output = self.output.numpy()
            flags = self.data.overflow.numpy()
            contacts = int(self.contact_peak.numpy()[0])
            constraints = self.constraint_peak.numpy()
            cursor = int(self.cursor.numpy()[0])
            transfer = time.perf_counter() - start
        iterations = int(self.mjw.OverflowType.ITERATIONS | self.mjw.OverflowType.LS_ITERATIONS)
        diagnostics = _capacity_diagnostics(
            flags, contacts, constraints, iteration_mask=iterations, naconmax=self.data.naconmax,
            njmax=self.njmax, cursor=cursor, steps=self.steps,
        )
        return output, elapsed, transfer, diagnostics

    def close(self):
        # The caller uses one process per case, which also releases CUDA pools.
        pass


def run_case(config: dict) -> dict:
    """Return a JSON-serializable checked row, never a speedup claim.

    Run cases in isolated processes: tutorial asset helpers bind robot-specific
    module globals, and a process boundary reliably releases CUDA allocations.
    Invalid configurations and backend failures return ``status='failed'``.
    """
    row = {"status": "failed", "backend": config.get("backend", "mujoco"),
           "robot": config.get("robot", "so101"), "worlds": config.get("worlds", 1),
           "samples": [], "warmup_samples": [], "timings": {}, "diagnostics": {}}
    replay = None
    start_case = time.perf_counter()
    try:
        config = _configuration(config)
        start = time.perf_counter()
        model, initial, initial_state, tape, phases, spec, workload = _prepare(config)
        row["workload"] = workload
        output_bytes = config["worlds"] * workload["physics_steps"] * len(initial_state) * (8 if config["backend"] == "mujoco" else 4)
        row["workload"]["trajectory_bytes"] = output_bytes
        if output_bytes > config["max_trajectory_mib"] * 1024**2:
            raise ValueError(
                f"Trajectory requires {output_bytes / 1024**2:.1f} MiB, above max_trajectory_mib="
                f"{config['max_trajectory_mib']}; reduce worlds, keeping the complete task"
            )
        row["timings"]["preparation_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        replay_type = _CPUReplay if config["backend"] == "mujoco" else _GPUReplay
        replay = replay_type(config, model, initial, initial_state, tape)
        row["device_info"] = replay.info
        row["timings"]["backend_setup_seconds"] = time.perf_counter() - start
        for warmup in (True, False):
            start_group = time.perf_counter()
            for repeat in range(config["warmups"] if warmup else config["repeats"]):
                states, elapsed, transfer, diagnostics = replay.episode()
                start = time.perf_counter()
                validation = validate_trajectory(states, model, spec, phases)
                checked = time.perf_counter() - start
                passed = bool(validation["passed"] and diagnostics["capacity_passed"])
                sample = {"repeat": repeat, "simulation_seconds": elapsed,
                          "output_transfer_seconds": transfer, "validation_seconds": checked,
                          "task_success_count": validation["task_success_count"],
                          "task_total_count": validation["task_total_count"],
                          "passed": passed, "validation": validation, "diagnostics": diagnostics}
                row["warmup_samples" if warmup else "samples"].append(sample)
                if not passed:
                    raise ValueError(f"{'Warmup' if warmup else 'Measured'} episode {repeat} failed task or capacity validation")
            if warmup:
                row["timings"]["warmup_seconds"] = time.perf_counter() - start_group
        row["status"] = "passed"
    except Exception as exc:
        row["diagnostics"]["error_type"] = type(exc).__name__
        row["diagnostics"]["error"] = str(exc)
    finally:
        if replay is not None:
            replay.close()
        row["timings"]["case_seconds"] = time.perf_counter() - start_case
    return row
