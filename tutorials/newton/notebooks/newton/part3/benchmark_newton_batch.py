# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Full-task Newton CPU process-pool versus Newton CUDA world batching.

Both paths call Newton's SolverMuJoCo.step with Newton State and Control.
Newton 1.6 native CPU advances one template world, so independent CPU worlds
run on persistent worker processes. CUDA uses ModelBuilder.replicate and one
batched solver. Precomputed targets exclude online IK from both timers.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import re
import sys
import time
import traceback

import mujoco
import numpy as np
import warp as wp
import newton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solutions"))
from newton_scene_solution import (  # noqa: E402
    build_ik_model, build_newton_builder, robot_joint_indices,
)
from robots import get_robot  # noqa: E402
import pick_place_common as task  # noqa: E402
from box_task import BoxTask  # noqa: E402
import benchmark_box_protocol as box_protocol  # noqa: E402

FRAMES, SUBSTEPS, DT = box_protocol.FRAMES, box_protocol.SUBSTEPS, box_protocol.DT
FIELDS = box_protocol.OBSERVATION_SIZE
_WORKER = None

# This isolated box task has one physics solver. MuJoCo owns the integrated
# coordinates between steps; reset() explicitly synchronizes the initial state.
# Coupled solvers or external state edits require a separate synchronization policy.
STATE_SYNC_POLICY = {
    "version": 1,
    "requested_update_data_interval": 0,
    "scope": "single SolverMuJoCo box batch, both native CPU and CUDA",
    "reason": "avoid redundant per-step Newton-to-MuJoCo coordinate round trips; no external solver edits state",
    "reset": "SolverMuJoCo.reset explicitly synchronizes initial joint coordinates when interval != 1",
    "controls_and_forces": "applied every step; MuJoCo-to-Newton output state remains updated every step",
}


@wp.kernel(enable_backward=False)
def _targets(tape: wp.array2d(dtype=float), indices: wp.array(dtype=int),
             cursor: wp.array(dtype=int), targets: wp.array(dtype=float), stride: int):
    world, actuator = wp.tid()
    targets[world * stride + indices[actuator]] = tape[cursor[0], actuator]


@wp.kernel(enable_backward=False)
def _advance(cursor: wp.array(dtype=int)):
    cursor[0] += 1


@wp.kernel(enable_backward=False)
def _diagnostics(poses: wp.array(dtype=wp.transform), velocities: wp.array(dtype=wp.spatial_vector),
                 bad: wp.array(dtype=int), bodies_per_world: int, nacon: wp.array(dtype=int),
                 nefc: wp.array(dtype=int), peak_contacts: wp.array(dtype=int),
                 peak_constraints: wp.array(dtype=int), step_counts: wp.array(dtype=int),
                 count_step: int, ncollision: wp.array(dtype=int),
                 peak_broadphase_pairs: wp.array(dtype=int)):
    world, local_body = wp.tid()
    body = world * bodies_per_world + local_body
    position = wp.transform_get_translation(poses[body])
    rotation = wp.transform_get_rotation(poses[body])
    for i in range(3):
        if not wp.isfinite(position[i]):
            wp.atomic_max(bad, world, 1)
    for i in range(4):
        if not wp.isfinite(rotation[i]):
            wp.atomic_max(bad, world, 1)
    for i in range(6):
        if not wp.isfinite(velocities[body][i]):
            wp.atomic_max(bad, world, 1)
    if local_body == 0:
        # Count completed solver calls independently of the control-frame
        # cursor. The observation-only forward pass passes count_step=0.
        step_counts[world] += count_step
        wp.atomic_max(peak_constraints, world, nefc[world])
        if world == 0:
            wp.atomic_max(peak_contacts, 0, nacon[0])
            # An observation-only forward can overflow broadphase before the
            # next integration updates sticky flags. Preserve its raw peak.
            wp.atomic_max(peak_broadphase_pairs, 0, ncollision[0])


def _configuration(config):
    result = {"scope": "newton-batch", "task": "box", "backend": "newton_cpu", "robot": "so101",
              "worlds": 1, "cpu_threads": 4, "frames": FRAMES, "substeps": SUBSTEPS,
              "warmups": 1, "repeats": 5, "device": "cuda:0", "max_trajectory_mib": 1024,
              "nconmax": None, "njmax": None, **config}
    if result["scope"] != "newton-batch" or result["backend"] not in ("newton_cpu", "newton_cuda"):
        raise ValueError("Expected newton-batch scope with newton_cpu or newton_cuda backend")
    if result["task"] != "box":
        raise ValueError("The Newton batch benchmark executes the two-cube receiving-box task")
    result.setdefault("validation_workers", result["cpu_threads"])
    get_robot(result["robot"])
    for key in ("worlds", "cpu_threads", "validation_workers", "frames", "substeps", "warmups", "repeats"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if result["frames"] != FRAMES or result["substeps"] != SUBSTEPS:
        raise ValueError("Every world must run all 2000 control frames and 40000 physics steps")
    for key in ("nconmax", "njmax"):
        if result[key] is not None and (type(result[key]) is not int or result[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    limit = float(result["max_trajectory_mib"])
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("max_trajectory_mib must be finite and positive")
    required = result["worlds"] * FRAMES * FIELDS * np.dtype(np.float32).itemsize
    if required > limit * 2**20:
        raise ValueError(f"Payload history needs {required / 2**20:.2f} MiB; cannot shorten the task to fit")
    if result["backend"] == "newton_cuda" and not str(result["device"]).startswith("cuda:"):
        raise ValueError("newton_cuda requires an explicit CUDA device; no CPU substitution")
    return result


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _newton_binding(model, spec):
    """Resolve imported names without assuming Newton preserves MJCF names.

    Newton appends a shape index to geom labels and flattens body paths. Match
    the authored terminal label, require uniqueness, and retain the actual
    compiled IDs used by the backend's solved contacts.
    """
    def geom_id(name):
        matches = [index for index in range(model.ngeom)
                   if re.search(r"(?:^|/)" + re.escape(name) + r"_\d+$", model.geom(index).name or "")]
        if len(matches) != 1:
            raise ValueError(f"Expected one compiled Newton geom for {name!r}; found {matches}")
        return matches[0]

    if spec.left_jaw_geoms:
        roots = [int(model.geom_bodyid[geom_id(group[0])])
                 for group in (spec.left_jaw_geoms, spec.right_jaw_geoms)]
    else:
        roots = []
        for name in (spec.left_jaw_body, spec.right_jaw_body):
            matches = [index for index in range(model.nbody)
                       if (model.body(index).name or "").endswith("_" + name)]
            if len(matches) != 1:
                raise ValueError(f"Expected one compiled Newton jaw body {name!r}; found {matches}")
            roots.append(matches[0])
    groups = []
    for root in roots:
        bodies = {root}
        for body in range(root + 1, model.nbody):
            if model.body_parentid[body] in bodies:
                bodies.add(body)
        groups.append({index for index in range(model.ngeom) if model.geom_bodyid[index] in bodies
                       and (model.geom_contype[index] or model.geom_conaffinity[index])})
    groups[0] -= groups[1]
    if not all(groups) or groups[0] & groups[1]:
        raise ValueError("Newton grasp evidence requires two nonempty, disjoint jaw collider groups")
    qpos = [int(model.jnt_qposadr[model.actuator_trnid[index, 0]]) for index in spec.gripper_ctrl_indices]
    return {"cube_geoms": [geom_id(name) for name in ("red_cube", "blue_cube")],
            "jaw_geoms": [sorted(group) for group in groups], "gripper_qpos": qpos}


def _prepare(config):
    spec = get_robot(config["robot"])
    box = BoxTask(spec)
    model, data, scene = build_ik_model(spec, task_name="box")
    spec = box.spec
    controller = box.make_controller()
    commands, phases = [], []
    for _ in range(FRAMES):
        phases.append(controller.phase_name())
        commands.append(controller.step(model, data, .02).copy())
    tape = np.asarray(commands, dtype=np.float32)
    if not controller.done or not np.isfinite(tape).all():
        raise ValueError("Precomputed full-task controller tape is incomplete or invalid")
    names = ("benchmark_newton_batch.py", "benchmark_box_protocol.py", "box_task.py",
             "pick_place_common.py", "robots.py", "utils.py", "solutions/newton_scene_solution.py")
    protocol = {
        "task": "two_cube_pick_place_into_box", "worlds": config["worlds"],
        "frames": FRAMES, "substeps": SUBSTEPS, "physics_steps": FRAMES * SUBSTEPS,
        "physics_dt_seconds": DT, "simulated_seconds": 40.0,
        "bin_lower": box.box.lower.tolist(), "bin_upper": box.box.upper.tolist(),
        "acceptance": box_protocol.ACCEPTANCE,
        "host_validation_policy": {**box_protocol.HOST_VALIDATION_POLICY,
            "requested_workers":config.get("validation_workers",config["cpu_threads"])},
        "controller": "identical precomputed float32 Newton joint-position targets",
        "initial_robot_state": "first above-red target, assigned before rollout; avoids home pose intersecting table/box",
        "initial_robot_ctrl": tape[0].tolist(),
        "initial_robot_ctrl_sha256": _sha(tape[0].astype('<f4').tobytes()),
        "state_sync_policy": dict(STATE_SYNC_POLICY),
        "contact_detection": "MuJoCo", "output": "61 float32 physical box-task observations per world at 50 Hz; live solved contacts",
        "timed_scope": "episode reset, Newton control updates, 40000 SolverMuJoCo.step calls, per-frame forward and live geometry/solved-contact recording; synchronized CUDA or CPU pool dispatch and completion",
        "source_sha256": {name: _sha((HERE / name).read_bytes()) for name in names},
        "scene_xml_sha256": _sha(scene.read_bytes()),
        "robot_xml_sha256": _sha((scene.parent / spec.robot_xml).read_bytes()),
        "asset_ref": spec.menagerie_ref, "control_tape_sha256": _sha(tape.astype('<f4').tobytes()),
        "solver_settings": {"solver": "newton", "integrator": "implicitfast", "cone": "elliptic",
                            "iterations": 100, "ls_iterations": 50, "impratio": 100,
                            "tolerance": 1e-6, "ls_tolerance": .01},
        "backend_policy": {
            "collision_detection": "backend defaults; no collider flag override",
            "native_clock_precision": "float64", "gpu_clock_precision": "float32",
            "numerical_options": "native requested and actual uploaded GPU options recorded separately",
        },
    }
    protocol["comparison_signature"] = _sha(json.dumps(protocol, sort_keys=True).encode())
    return tape, np.asarray(phases), spec, protocol


def _build(config, worlds, initial_ctrl):
    spec = get_robot(config["robot"])
    template = build_newton_builder(spec, task_name="box", initial_ctrl=initial_ctrl)
    ik_model, _, _ = build_ik_model(spec, task_name="box")
    joints = robot_joint_indices(template, ik_model)
    red, blue = [next(i for i, name in enumerate(template.body_label) if name.endswith(label))
                 for label in ("red_cube", "blue_cube")]
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(template, worlds)
    model = builder.finalize()
    target_indices = model.joint_target_q_start.numpy()[joints]
    cpu = config["backend"] == "newton_cpu"
    if cpu and worlds != 1:
        raise ValueError("Native Newton CPU solver owns one template world; use independent worker processes")
    solver = newton.solvers.SolverMuJoCo(
        model, use_mujoco_cpu=cpu, separate_worlds=True, use_mujoco_contacts=True,
        solver="newton", integrator="implicitfast", cone="elliptic", impratio=100,
        iterations=100, ls_iterations=50, tolerance=1e-6, update_data_interval=0,
        nconmax=config["nconmax"] or spec.nconmax,
        njmax=config["njmax"] or max(spec.njmax, 1024),
    )
    states = (model.state(), model.state())
    solver.mj_model.opt.timestep = DT
    if not cpu:
        # Record the actual execution timestep before warmup/capture; step()
        # subsequently supplies this same dt on every integration call.
        solver.mjw_model.opt.timestep.fill_(DT)
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    if model.world_count != worlds:
        raise ValueError("Newton model world count does not match requested batch")
    buffer = np.empty(mujoco.mj_sizeModel(solver.mj_model), dtype=np.uint8)
    mujoco.mj_saveModel(solver.mj_model, buffer=buffer)
    metadata = {"model_mjb_sha256": _sha(buffer.tobytes()),
                "initial_compiled_qpos0_sha256": _sha(solver.mj_model.qpos0.astype('<f8').tobytes()),
                "initial_newton_joint_q_first_world_sha256": _sha(
                    model.joint_q.numpy()[:model.joint_q.size // worlds].astype('<f4').tobytes()),
                "initial_newton_joint_qd_first_world_sha256": _sha(
                    model.joint_qd.numpy()[:model.joint_qd.size // worlds].astype('<f4').tobytes()),
                "model_dimensions": {key: int(getattr(solver.mj_model, key))
                                     for key in ("nq", "nv", "nu", "nbody", "ngeom")},
                "solver_options": {key: getattr(solver.mj_model.opt, key) for key in
                                   ("timestep", "iterations", "ls_iterations", "impratio",
                                    "tolerance", "ls_tolerance")},
                "backend_settings": box_protocol.backend_option_metadata(
                    solver.mj_model, None if cpu else solver.mjw_model)}
    metadata["backend_settings"]["newton_state_sync"] = {
        **STATE_SYNC_POLICY,
        "actual_update_data_interval": int(solver.update_data_interval),
        "enable_sleeping": bool(solver.enable_sleeping),
    }
    if solver.update_data_interval != 0 or solver.enable_sleeping:
        raise ValueError("Box benchmark requires persistent MuJoCo coordinates and explicit non-sleeping reset")
    metadata["backend_settings"]["active_backend"] = "native_mujoco" if cpu else "mujoco_warp"
    metadata["backend_settings"]["active_clock_precision"] = "float64" if cpu else "float32"
    return model, solver, states, model.control(), target_indices, red, blue, metadata


class _CPUWorld:
    def __init__(self, config, tape):
        wp.init()
        wp.set_device("cpu")
        (self.model, self.solver, self.states, self.control, self.indices,
         self.red, self.blue, self.metadata) = _build(config, 1, tape[0])
        self.tape = tape
        self.targets = self.control.joint_target_q.numpy().copy()
        spec = BoxTask(get_robot(config["robot"])).spec
        self.recorder = box_protocol.NativeRecorder(self.solver.mj_model, spec,
                                                    **_newton_binding(self.solver.mj_model, spec))

    def episode(self, output):
        model, solver = self.model, self.solver
        state, other = self.states
        mujoco.mj_resetData(solver.mj_model, solver.mj_data)
        solver.reset(state)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        finite = True
        step_count = 0
        for frame in range(FRAMES):
            self.targets[self.indices] = self.tape[frame]
            self.control.joint_target_q.assign(self.targets)
            for _ in range(SUBSTEPS):
                state.clear_forces()
                solver.step(state, other, self.control, None, DT)
                step_count += 1
                state, other = other, state
                finite = finite and bool(np.isfinite(solver.mj_data.qpos).all()
                                         and np.isfinite(solver.mj_data.qvel).all())
            # MuJoCo's step ends after integration; refresh the solved contacts
            # and geometry at the observed state, identically on both devices.
            mujoco.mj_forward(solver.mj_model, solver.mj_data)
            self.recorder.capture(solver.mj_data, output[frame], frame)
        warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(solver.mj_data.warning) if w.number}
        return {"finite_all_bodies": finite, "warnings": warnings, "recorded_physics_steps": step_count,
                "final_time_seconds": float(solver.mj_data.time),
                "capacity_passed": bool(finite and not warnings and step_count == FRAMES * SUBSTEPS)}


def _cpu_initialize(config, tape, shared_name, shape, ready, build_lock):
    global _WORKER
    try:
        # The tutorial resolves generated MJCF files during construction. Keep
        # those writes and shared JIT-cache setup serialized, outside the timer.
        with build_lock:
            world = _CPUWorld(config, tape)
        memory = shared_memory.SharedMemory(name=shared_name)
        output = np.ndarray(shape, dtype=np.float32, buffer=memory.buf)
        _WORKER = (world, memory, output)
        ready.put({"passed": True, "metadata": world.metadata})
    except Exception as error:
        ready.put({"passed": False, "error": f"{type(error).__name__}: {error}"})
        raise


def _cpu_episode(world):
    context, _, output = _WORKER
    return {"world": world, **context.episode(output[world])}


class _CPUBatch:
    def __init__(self, config, tape):
        self.worlds = config["worlds"]
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        self.workers = min(self.worlds, config["cpu_threads"], affinity)
        shape = (self.worlds, FRAMES, FIELDS)
        required = int(np.prod(shape)) * 4
        if sys.platform.startswith("linux") and Path("/dev/shm").exists():
            available = os.statvfs("/dev/shm")
            if required > available.f_bavail * available.f_frsize:
                raise ValueError("Insufficient /dev/shm space for CPU histories; increase shared memory without shortening the task")
        self.memory = shared_memory.SharedMemory(create=True, size=required)
        self.output = np.ndarray(shape, dtype=np.float32, buffer=self.memory.buf)
        self.pool = None
        # Avoid inheriting initialized Warp/CUDA state through fork. Each CPU
        # worker owns its Newton objects; only the disjoint output rows are shared.
        ctx = mp.get_context("spawn")
        self.ready = ctx.Queue()
        overrides = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "PXR_WORK_THREAD_LIMIT": "1"}
        prior = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            self.pool = ctx.Pool(self.workers, initializer=_cpu_initialize,
                                 initargs=(config, tape, self.memory.name, shape, self.ready, ctx.Lock()))
            initialized = [self.ready.get(timeout=300) for _ in range(self.workers)]
            if not all(row["passed"] for row in initialized):
                raise RuntimeError(f"CPU Newton initialization failed: {initialized}")
            self.metadata = initialized[0]["metadata"]
            if any(row["metadata"] != self.metadata for row in initialized):
                raise ValueError("Independent CPU workers constructed different Newton models")
        except BaseException:
            self.close()
            raise
        finally:
            for key, value in prior.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def episode(self):
        start = time.perf_counter()
        diagnostics = self.pool.map(_cpu_episode, range(self.worlds), chunksize=1)
        elapsed = time.perf_counter() - start
        start = time.perf_counter()
        output = self.output.copy()
        transfer = time.perf_counter() - start
        return output, elapsed, transfer, {
            "capacity_passed": all(row["capacity_passed"] for row in diagnostics),
            "worlds": diagnostics, "completed_worlds": len(diagnostics),
        }

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
        self.memory.close()
        self.memory.unlink()
        self.ready.close()


class _CUDABatch:
    def __init__(self, config, tape):
        import mujoco_warp as mjw
        self.mjw = mjw
        wp.init()
        self.device = wp.get_device(config["device"])
        if not self.device.is_cuda:
            raise ValueError("Newton CUDA benchmark requires CUDA; no CPU substitution")
        self.worlds, self.workers = config["worlds"], 0
        with wp.ScopedDevice(self.device):
            (self.model, self.solver, self.states, self.control, indices,
             self.red, self.blue, self.metadata) = _build(config, self.worlds, tape[0])
            self.data = self.solver.mjw_data
            if self.data.nworld != self.worlds:
                raise ValueError("Solver did not allocate every requested world")
            self.bodies = self.model.body_count // self.worlds
            self.target_stride = self.control.joint_target_q.size // self.worlds
            self.indices = wp.array(indices, dtype=int)
            self.tape = wp.array(tape, dtype=float)
            self.cursor = wp.zeros(1, dtype=int)
            spec = BoxTask(get_robot(config["robot"])).spec
            self.recorder = box_protocol.CUDARecorder(self.solver.mj_model, self.solver.mjw_model,
                self.data, spec, **_newton_binding(self.solver.mj_model, spec))
            self.output = self.recorder.output
            self.bad = wp.zeros(self.worlds, dtype=int)
            self.peak_contacts = wp.zeros(1, dtype=int)
            self.peak_constraints = wp.zeros(self.worlds, dtype=int)
            self.step_counts = wp.zeros(self.worlds, dtype=int)
            self.peak_broadphase_pairs = wp.zeros(1, dtype=int)
            # Build kernels before capture and then reset the consumed frame.
            self.reset()
            self.frame()
            self.reset()
            with wp.ScopedCapture() as capture:
                self.frame()
            self.graph = capture.graph
            self.reset()

    def reset(self):
        self.mjw.reset_data(self.solver.mjw_model, self.data)
        self.solver.reset(self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        self.cursor.zero_()
        self.bad.zero_()
        self.peak_contacts.zero_()
        self.peak_constraints.zero_()
        self.step_counts.zero_()
        self.peak_broadphase_pairs.zero_()

    def frame(self):
        wp.launch(_targets, dim=(self.worlds, self.indices.size), inputs=[
            self.tape, self.indices, self.cursor, self.control.joint_target_q, self.target_stride])
        state, other = self.states
        for _ in range(SUBSTEPS):
            state.clear_forces()
            self.solver.step(state, other, self.control, None, DT)
            state, other = other, state
            wp.launch(_diagnostics, dim=(self.worlds, self.bodies), inputs=[
                state.body_q, state.body_qd, self.bad, self.bodies, self.data.nacon, self.data.nefc,
                self.peak_contacts, self.peak_constraints, self.step_counts, 1,
                self.data.ncollision, self.peak_broadphase_pairs])
        self.mjw.forward(self.solver.mjw_model, self.data)
        wp.launch(_diagnostics, dim=(self.worlds, self.bodies), inputs=[
            state.body_q, state.body_qd, self.bad, self.bodies, self.data.nacon, self.data.nefc,
            self.peak_contacts, self.peak_constraints, self.step_counts, 0,
            self.data.ncollision, self.peak_broadphase_pairs])
        self.recorder.record(self.cursor)
        wp.launch(_advance, dim=1, inputs=[self.cursor])

    def episode(self):
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            start = time.perf_counter()
            self.reset()
            for _ in range(FRAMES):
                wp.capture_launch(self.graph)
            wp.synchronize_device(self.device)
            elapsed = time.perf_counter() - start
            start = time.perf_counter()
            output = self.output.numpy()
            flags = self.data.overflow.numpy()
            bad = self.bad.numpy()
            cursor = int(self.cursor.numpy()[0])
            contacts = int(self.peak_contacts.numpy()[0])
            constraints = self.peak_constraints.numpy()
            step_counts = self.step_counts.numpy()
            broadphase_pairs = int(self.peak_broadphase_pairs.numpy()[0])
            transfer = time.perf_counter() - start
        iterations = int(self.mjw.OverflowType.ITERATIONS | self.mjw.OverflowType.LS_ITERATIONS)
        capacity = flags & ~iterations
        return output, elapsed, transfer, {
            "capacity_passed": bool(cursor == FRAMES and np.all(step_counts == FRAMES * SUBSTEPS)
                                    and not np.any(capacity) and not np.any(bad)
                                    and contacts <= self.data.naconmax
                                    and broadphase_pairs <= self.data.naconmax
                                    and np.all(constraints <= self.data.njmax)),
            "overflow_flags_per_world": flags.tolist(), "capacity_flags_per_world": capacity.tolist(),
            "iteration_warning_flags_per_world": (flags & iterations).tolist(),
            "nonfinite_state_per_world": bad.tolist(), "recorded_control_frames": cursor,
            "recorded_physics_steps_per_world": step_counts.tolist(),
            "frame_cursor_derived_physics_steps": cursor * SUBSTEPS,
            "peak_contacts_all_worlds": contacts, "peak_constraints_per_world": constraints.tolist(),
            "peak_broadphase_pairs_all_worlds": broadphase_pairs,
            "naconmax_all_worlds": int(self.data.naconmax), "njmax_per_world": int(self.data.njmax),
        }

    def close(self):
        pass  # Case worker exit releases device allocations.


def run_case(config):
    started = time.perf_counter()
    row = {"status": "failed", "scope": "newton-batch", "samples": [], "warmup_samples": [],
           "config": dict(config), "workload": {}, "timings": {}}
    engine = validator = None
    try:
        config = _configuration(config)
        row.update(config=config, backend=config["backend"], robot=config["robot"], worlds=config["worlds"])
        tape, phases, spec, row["workload"] = _prepare(config)
        row["timings"]["preparation_seconds"] = time.perf_counter() - started
        setup = time.perf_counter()
        engine = (_CPUBatch if config["backend"] == "newton_cpu" else _CUDABatch)(config, tape)
        row["timings"]["backend_setup_seconds"] = time.perf_counter() - setup
        row["workload"].update(engine.metadata)
        row["device_info"] = {
            "device": "cpu" if config["backend"] == "newton_cpu" else config["device"],
            "cpu_threads": engine.workers, "cpu_workers": engine.workers,
            "cpu_parallelism": "persistent processes, one single-world Newton solver per worker",
            "gpu_worlds": config["worlds"] if config["backend"] == "newton_cuda" else 0,
            "cpu_worker_environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                       "PXR_WORK_THREAD_LIMIT": "1"} if config["backend"] == "newton_cpu" else None,
        }
        setup = time.perf_counter()
        validator = box_protocol.ObservationValidator(spec, phases, config["worlds"], config["validation_workers"])
        row["timings"]["validation_setup_seconds"] = time.perf_counter()-setup
        row["host_validation"] = validator.info
        warmup_seconds = 0.0
        for repeat in range(config["warmups"] + config["repeats"]):
            history, elapsed, transfer, diagnostics = engine.episode()
            validate_started = time.perf_counter()
            if config["backend"] == "newton_cpu":
                counts = {world["world"]: world["recorded_physics_steps"]
                          for world in diagnostics["worlds"]}
                step_counts = [counts.get(world) for world in range(config["worlds"])]
                clock_precision = "float64"
            else:
                step_counts = diagnostics["recorded_physics_steps_per_world"]
                clock_precision = "float32"
            validation = validator.validate(
                history, clock_precision=clock_precision, step_counts=step_counts)
            validate_seconds = time.perf_counter() - validate_started
            passed = bool(validation["passed"] and diagnostics["capacity_passed"])
            sample = {"simulation_seconds": elapsed, "output_transfer_seconds": transfer,
                      "validation_seconds": validate_seconds, "passed": passed,
                      "task_success_count": validation["task_success_count"],
                      "task_total_count": config["worlds"], "validation": validation,
                      "diagnostics": diagnostics}
            if repeat < config["warmups"]:
                row["warmup_samples"].append(sample)
                warmup_seconds += elapsed + transfer + validate_seconds
            else:
                sample["repeat"] = repeat - config["warmups"]
                row["samples"].append(sample)
            if not passed:
                raise ValueError("Full Newton batch failed task, time progression, finite-state or capacity checks")
        row["timings"]["warmup_seconds"] = warmup_seconds
        row["status"] = "passed"
    except Exception as error:
        row.update(error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        close_started = time.perf_counter()
        try:
            if validator is not None:
                validator.close()
        finally:
            row["timings"]["validation_teardown_seconds"] = time.perf_counter()-close_started
            try:
                if engine is not None:
                    engine.close()
            finally:
                row["timings"]["case_seconds"] = time.perf_counter() - started
    return row
