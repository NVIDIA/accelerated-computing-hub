# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Common measured two-cube box task; no physics or inferred contact forces.

Each frame stores [time, actual opening, final-frame jaw clearance], followed
by two blocks of [8 world corners, maximum corner speed, 2 jaw contact counts,
2 solved normal forces]. The 61 float32 fields are identical on CPU and CUDA.
Jaw clearance is needed only at the final frame; earlier entries are zero.
"""
from __future__ import annotations

from itertools import product
from functools import lru_cache
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import warp as wp

from box_task import BoxTask, CubeEvidence, NAMES, cube_points, geom_lower_z, native_contacts
from robots import get_robot

FRAMES, SUBSTEPS, DT = 2000, 20, .001
OBSERVATION_SIZE = 61
ACCEPTANCE = {
    "protocol_version": 2,
    "clock_validation": "every raw float32 observation matches sequential backend-precision integration within one observation ULP; independent per-world integration count",
    "clock_observation_ulp_allowance": 1,
    "task": "two_cube_pick_place_into_box", "frames": FRAMES, "physics_steps": FRAMES * SUBSTEPS,
    "simulated_seconds": 40.0, "control_hz": 50,
    "bilateral_close_frames": 10, "minimum_jaw_normal_force_N": 1e-5,
    "whole_cube_lift_m": .01, "consecutive_loaded_lift_frames": 25,
    "loaded_carry_frames": 10, "horizontal_carry_m": .05,
    "whole_cube_box_tolerance_m": .003, "release_open_fraction": .8,
    "detached_settled_frames": 50, "maximum_settled_corner_speed_m_s": .04,
    "final_jaw_clearance_m": .02, "ordered_cube_cycles": list(NAMES),
    "observation_fields": OBSERVATION_SIZE, "observation_dtype": "float32",
    "jaw_clearance_observation": "final control frame only; earlier entries are zero",
    "contact_observation": "actual solved backend contact forces after a backend forward pass",
}


def _base_spec(spec):
    return get_robot(spec) if isinstance(spec, str) else spec


def bind(model, spec, *, cube_geoms=None, jaw_geoms=None, gripper_qpos=None):
    """Bind native or Newton-compiled model IDs; overrides never change gates."""
    box = BoxTask(_base_spec(spec))
    if cube_geoms is None or jaw_geoms is None:
        box._bind(model)
    cubes = np.asarray(cube_geoms if cube_geoms is not None else [box._geom_ids[n] for n in NAMES], dtype=np.int32)
    jaws = [sorted(set(int(g) for g in group)) for group in (jaw_geoms if jaw_geoms is not None else box._jaw_geoms)]
    if len(cubes) != 2 or len(jaws) != 2 or not all(jaws) or set(jaws[0]) & set(jaws[1]):
        raise ValueError("Two distinct cubes and two disjoint nonempty jaw groups are required")
    if len(set(cubes)) != 2 or any(not 0 <= g < model.ngeom for g in [*cubes, *jaws[0], *jaws[1]]):
        raise ValueError("Invalid cube or jaw geometry binding")
    addresses = np.asarray(gripper_qpos if gripper_qpos is not None else [
        model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in box.spec.gripper_ctrl_indices], dtype=np.int32)
    if not len(addresses) or np.any(addresses < 0) or np.any(addresses >= model.nq):
        raise ValueError("Actual gripper joint position addresses are required")
    for geom in cubes:
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Payload evidence requires box collision geometries")
    supported = {int(x) for x in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_BOX)}
    if any(int(model.geom_type[g]) not in supported for group in jaws for g in group):
        raise ValueError("Unsupported jaw collision geometry")
    return {"cube_geoms": cubes, "jaw_geoms": jaws, "gripper_qpos": addresses,
            "cube_bodies": np.asarray(model.geom_bodyid[cubes], dtype=np.int32),
            "box": box}


class NativeRecorder:
    def __init__(self, model, spec, **overrides):
        self.model = model
        self.binding = bind(model, spec, **overrides)
        self.box = self.binding["box"]
        self.signs = np.asarray(list(product((-1., 1.), repeat=3)))

    def capture(self, data, output, frame):
        """Record already-forwarded live data, without recomputing its contacts."""
        model, binding = self.model, self.binding
        if output.shape != (OBSERVATION_SIZE,):
            raise ValueError("Expected one 61-field observation")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite native physics state")
        output.fill(0)
        output[0] = data.time
        actual = np.mean(data.qpos[binding["gripper_qpos"]])
        output[1] = np.clip((actual-self.box.spec.gripper_closed) /
                            (self.box.spec.gripper_open-self.box.spec.gripper_closed), 0, 1)
        contacts = native_contacts(model, data)
        for cube, geom in enumerate(binding["cube_geoms"]):
            start = 3 + cube * 29
            points = (self.signs * model.geom_size[geom]) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            body = int(binding["cube_bodies"][cube])
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            speeds = velocity[3:] + np.cross(velocity[:3], points-data.xpos[body])
            output[start:start+24] = points.ravel()
            output[start+24] = np.linalg.norm(speeds, axis=1).max()
            for first, second, force in contacts:
                if geom not in (first, second):
                    continue
                other = second if first == geom else first
                for jaw, geoms in enumerate(binding["jaw_geoms"]):
                    if other in geoms:
                        output[start+25+jaw] += 1
                        output[start+27+jaw] += force
        if frame == FRAMES-1:
            output[2] = min(geom_lower_z(model, data, g) for group in binding["jaw_geoms"] for g in group)
            output[2] -= max(self.box.spec.table_top_z, self.box.box.upper[2])


@wp.kernel(enable_backward=False)
def _geometry(qpos: wp.array2d(dtype=float), geom_pos: wp.array2d(dtype=wp.vec3),
              geom_mat: wp.array2d(dtype=wp.mat33), cvel: wp.array2d(dtype=wp.spatial_vector),
              subtree_com: wp.array2d(dtype=wp.vec3), sim_time: wp.array(dtype=float),
              cubes: wp.array(dtype=int), bodies: wp.array(dtype=int), roots: wp.array(dtype=int),
              sizes: wp.array(dtype=wp.vec3), signs: wp.array(dtype=wp.vec3),
              grip_qpos: wp.array(dtype=int), closed: float, opened: float,
              cursor: wp.array(dtype=int), history: wp.array3d(dtype=float)):
    world, cube = wp.tid()
    frame = cursor[0]
    if cube == 0:
        history[world, frame, 0] = sim_time[world]
        actual = float(0.)
        for i in range(grip_qpos.shape[0]):
            actual += qpos[world, grip_qpos[i]]
        actual /= float(grip_qpos.shape[0])
        history[world, frame, 1] = wp.clamp((actual-closed)/(opened-closed), 0., 1.)
        history[world, frame, 2] = 0.
    start = 3 + cube * 29
    geom, body = cubes[cube], bodies[cube]
    velocity = cvel[world, body]
    angular = wp.vec3(velocity[0], velocity[1], velocity[2])
    linear = wp.vec3(velocity[3], velocity[4], velocity[5])
    maximum = float(0.)
    for corner in range(8):
        point = geom_mat[world, geom] @ wp.cw_mul(signs[corner], sizes[cube]) + geom_pos[world, geom]
        for axis in range(3):
            history[world, frame, start+corner*3+axis] = point[axis]
        point_velocity = linear + wp.cross(angular, point-subtree_com[world, roots[cube]])
        maximum = wp.max(maximum, wp.length(point_velocity))
    history[world, frame, start+24] = maximum
    for field in range(4):
        history[world, frame, start+25+field] = 0.


@wp.kernel(enable_backward=False)
def _contact_observations(nacon: wp.array(dtype=int), pairs: wp.array(dtype=wp.vec2i),
                          worlds: wp.array(dtype=int), forces: wp.array(dtype=wp.spatial_vector),
                          cubes: wp.array(dtype=int), jaw_lookup: wp.array(dtype=int),
                          cursor: wp.array(dtype=int), history: wp.array3d(dtype=float)):
    contact = wp.tid()
    if contact < nacon[0]:
        pair = pairs[contact]
        world, frame = worlds[contact], cursor[0]
        for cube in range(2):
            other = int(-1)
            if pair[0] == cubes[cube]:
                other = pair[1]
            elif pair[1] == cubes[cube]:
                other = pair[0]
            if other >= 0 and other < jaw_lookup.shape[0]:
                jaw = jaw_lookup[other]
                if jaw >= 0:
                    force = forces[contact][0]
                    wp.atomic_add(history, world, frame, 3+cube*29+25+jaw, 1.)
                    # NaNs must propagate to the common finite-state rejection.
                    if not wp.isfinite(force):
                        wp.atomic_add(history, world, frame, 3+cube*29+27+jaw, force)
                    else:
                        wp.atomic_add(history, world, frame, 3+cube*29+27+jaw, wp.max(0., force))


@wp.kernel(enable_backward=False)
def _jaw_clearance(geom_pos: wp.array2d(dtype=wp.vec3), geom_mat: wp.array2d(dtype=wp.mat33),
                   jaws: wp.array(dtype=int), types: wp.array(dtype=int), sizes: wp.array(dtype=wp.vec3),
                   mesh_start: wp.array(dtype=int), mesh_count: wp.array(dtype=int),
                   vertices: wp.array(dtype=wp.vec3), threshold: float,
                   cursor: wp.array(dtype=int), history: wp.array3d(dtype=float)):
    world = wp.tid()
    frame = cursor[0]
    if frame == 1999:
        lowest = float(1.e20)
        for i in range(jaws.shape[0]):
            geom = jaws[i]
            center = geom_pos[world, geom]
            rotation = geom_mat[world, geom]
            row = wp.vec3(rotation[2, 0], rotation[2, 1], rotation[2, 2])
            size, kind = sizes[i], types[i]
            lower = float(center[2])
            if kind == 7:  # mesh, exact collider vertices
                lower = float(1.e20)
                for vertex in range(mesh_start[i], mesh_start[i]+mesh_count[i]):
                    lower = wp.min(lower, wp.dot(row, vertices[vertex])+center[2])
            elif kind == 2:  # sphere
                lower -= size[0]
            elif kind == 3:  # capsule
                lower -= size[0]+wp.abs(row[2])*size[1]
            elif kind == 5:  # cylinder
                lower -= size[0]*wp.sqrt(row[0]*row[0]+row[1]*row[1])+wp.abs(row[2])*size[1]
            elif kind == 4:  # ellipsoid
                lower -= wp.length(wp.cw_mul(row, size))
            elif kind == 6:  # box
                lower -= wp.dot(wp.vec3(wp.abs(row[0]), wp.abs(row[1]), wp.abs(row[2])), size)
            lowest = wp.min(lowest, lower)
        history[world, frame, 2] = lowest-threshold


class CUDARecorder:
    """Graph-safe MuJoCo Warp observations from live device geometry and forces."""
    def __init__(self, native_model, mjw_model, mjw_data, spec, **overrides):
        import mujoco_warp as mjw
        self.mjw, self.model, self.data = mjw, mjw_model, mjw_data
        b = self.binding = bind(native_model, spec, **overrides)
        self.box = b["box"]
        self.output = wp.zeros((mjw_data.nworld, FRAMES, OBSERVATION_SIZE), dtype=float)
        self.cubes = wp.array(b["cube_geoms"], dtype=int)
        self.bodies = wp.array(b["cube_bodies"], dtype=int)
        self.roots = wp.array(native_model.body_rootid[b["cube_bodies"]], dtype=int)
        self.sizes = wp.array(native_model.geom_size[b["cube_geoms"]], dtype=wp.vec3)
        self.signs = wp.array(np.asarray(list(product((-1., 1.), repeat=3))), dtype=wp.vec3)
        self.grip_qpos = wp.array(b["gripper_qpos"], dtype=int)
        lookup = np.full(native_model.ngeom, -1, dtype=np.int32)
        for jaw, geoms in enumerate(b["jaw_geoms"]):
            lookup[geoms] = jaw
        self.jaw_lookup = wp.array(lookup, dtype=int)
        self.contact_ids = wp.array(np.arange(mjw_data.naconmax, dtype=np.int32), dtype=int)
        self.contact_forces = wp.zeros(mjw_data.naconmax, dtype=wp.spatial_vector)
        jaw_ids = np.asarray(sorted(set(b["jaw_geoms"][0]+b["jaw_geoms"][1])), dtype=np.int32)
        self.jaws = wp.array(jaw_ids, dtype=int)
        self.jaw_types = wp.array(native_model.geom_type[jaw_ids], dtype=int)
        self.jaw_sizes = wp.array(native_model.geom_size[jaw_ids], dtype=wp.vec3)
        vertices, starts, counts = [], [], []
        for geom in jaw_ids:
            starts.append(len(vertices))
            if native_model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = native_model.geom_dataid[geom]
                first, count = native_model.mesh_vertadr[mesh], native_model.mesh_vertnum[mesh]
                vertices.extend(native_model.mesh_vert[first:first+count])
                counts.append(count)
            else:
                counts.append(0)
        self.vertices = wp.array(np.asarray(vertices, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3)
        self.mesh_start, self.mesh_count = wp.array(starts, dtype=int), wp.array(counts, dtype=int)

    def record(self, cursor):
        d, box = self.data, self.box
        wp.launch(_geometry, dim=(d.nworld, 2), inputs=[d.qpos, d.geom_xpos, d.geom_xmat,
            d.cvel, d.subtree_com, d.time, self.cubes, self.bodies, self.roots, self.sizes,
            self.signs, self.grip_qpos, box.spec.gripper_closed, box.spec.gripper_open,
            cursor, self.output])
        self.mjw.contact_force(self.model, d, self.contact_ids, False, self.contact_forces)
        wp.launch(_contact_observations, dim=d.naconmax, inputs=[d.nacon, d.contact.geom,
            d.contact.worldid, self.contact_forces, self.cubes, self.jaw_lookup, cursor, self.output])
        wp.launch(_jaw_clearance, dim=d.nworld, inputs=[d.geom_xpos, d.geom_xmat, self.jaws,
            self.jaw_types, self.jaw_sizes, self.mesh_start, self.mesh_count, self.vertices,
            float(max(box.spec.table_top_z, box.box.upper[2])), cursor, self.output])


@lru_cache(maxsize=2)
def expected_clock(clock_precision):
    """Predict actual sequential additions, then the recorded float32 conversion.

    This deliberately does not substitute nominal frame times for raw time.
    Native MjData.time is float64; MuJoCo Warp time is float32. Both are
    recorded in the same float32 observation format after each control frame.
    """
    if clock_precision not in ("float32", "float64"):
        raise ValueError("Clock precision must describe the actual backend time storage")
    dtype = np.dtype(clock_precision).type
    current, timestep = dtype(0), dtype(DT)
    times = np.empty(FRAMES, dtype=np.float32)
    for frame in range(FRAMES):
        for _ in range(SUBSTEPS):
            current = dtype(current + timestep)
        times[frame] = current
    times.flags.writeable = False
    return times


def validate_clock(times, clock_precision, step_count):
    """Reject missing/duplicate integration, frozen clocks and wrong timesteps."""
    expected = expected_clock(clock_precision)
    times = np.asarray(times)
    if times.shape != (FRAMES,):
        raise ValueError("Clock validation requires every raw control-frame timestamp")
    finite = bool(np.isfinite(times).all())
    error = np.abs(times.astype(np.float64)-expected.astype(np.float64))
    ulp = np.spacing(expected).astype(np.float64)
    matching = np.isfinite(error) & (error <= ulp)
    count_ok = (isinstance(step_count, (int, np.integer))
                and not isinstance(step_count, (bool, np.bool_))
                and int(step_count) == FRAMES*SUBSTEPS)
    passed = bool(finite and count_ok and np.all(matching) and np.all(np.diff(times)>0))
    return {"passed": passed, "backend_clock_precision": clock_precision,
            "recorded_clock_precision": "float32", "allowed_observation_ulps": 1,
            "integration_count_passed": bool(count_ok),
            "recorded_physics_steps": int(step_count) if isinstance(step_count, (int, np.integer)) else None,
            "expected_physics_steps": FRAMES*SUBSTEPS,
            "mismatched_frame_count": int(np.count_nonzero(~matching)),
            "maximum_observation_ulp_error": float(np.max(error/ulp)) if finite else None,
            "expected_final_raw_time_s": float(expected[-1]),
            "nominal_integrated_duration_s": FRAMES*SUBSTEPS*DT}


def backend_option_metadata(native_model, uploaded_model=None):
    """Snapshot actual options outside timing; authored and uploaded may differ."""
    fields = ("timestep", "tolerance", "ls_tolerance", "ccd_tolerance", "impratio",
              "impratio_invsqrt", "iterations", "ls_iterations", "ccd_iterations",
              "solver", "integrator", "cone", "disableflags", "enableflags")

    def snapshot(options):
        result = {}
        for name in fields:
            if not hasattr(options, name):
                continue
            value = getattr(options, name)
            if hasattr(value, "numpy"):
                array = np.asarray(value.numpy())
                item = {"shape": list(array.shape), "dtype": str(array.dtype)}
                if array.size and np.all(array == array.flat[0]):
                    item["uniform_value"] = array.flat[0].item()
                else:
                    item["values"] = array.tolist()
                result[name] = item
            elif isinstance(value, (int, np.integer)):
                result[name] = int(value)
            else:
                result[name] = float(value)
        return result

    return {"native_requested": snapshot(native_model.opt),
            "uploaded_gpu_effective": snapshot(uploaded_model.opt) if uploaded_model is not None else None,
            "collision_policy": "backend defaults; no collider flag override",
            "note": "MuJoCo Warp may clamp requested tolerances for float32; actual uploaded values are recorded separately"}


def validate_observations(history, spec, phases, *, clock_precision, step_counts):
    """Replay the unchanged canonical CubeEvidence gates for every measured world."""
    history = np.asarray(history)
    if history.ndim != 3 or history.shape[1:] != (FRAMES, OBSERVATION_SIZE):
        raise ValueError("Box validation requires the complete 2000-frame, 61-field history")
    if len(phases) != FRAMES or phases[-1] != "done":
        raise ValueError("Box control sequence must finish before the end of the complete episode")
    if len(step_counts) != len(history):
        raise ValueError("An independently measured integration count is required for every world")
    spec = _base_spec(spec)
    results = []
    signs = np.asarray(list(product((-1., 1.), repeat=3)))
    for world, frames in enumerate(history):
        box = BoxTask(spec)
        finite = bool(np.isfinite(frames).all())
        times = np.asarray(frames[:, 0], dtype=np.float64)
        clock = validate_clock(times, clock_precision, step_counts[world])
        time_ok = bool(finite and clock["passed"])
        error = None
        if finite and time_ok:
            box.evidence = {name: CubeEvidence(name, box.box, box.spec.table_top_z,
                signs*box.spec.cube_half+position, 50)
                for name,position in zip(NAMES, (box.spec.red_cube_pos, box.spec.blue_cube_pos))}
            try:
                for index, (observation, phase) in enumerate(zip(frames, phases)):
                    active, _, command = str(phase).partition(":")
                    for cube, name in enumerate(NAMES):
                        start = 3+cube*29
                        box.evidence[name].observe(frame=index+1, time=(index+1)/50,
                            phase=command if active==name else "inactive",
                            points=observation[start:start+24].reshape(8, 3),
                            speed=float(observation[start+24]), counts=observation[start+25:start+27],
                            forces=observation[start+27:start+29], opening=float(observation[1]))
                box.frames, box.phase, box.tool_clearance = FRAMES, str(phases[-1]), float(frames[-1, 2])
            except ValueError as exc:
                error = str(exc)
        report = box.report()
        passed = bool(finite and time_ok and error is None and report["success"])
        results.append({"world": world, "passed": passed, "finite": finite,
                        "time_progression": time_ok, "clock": clock, "final_time_s": float(times[-1]) if finite else None,
                        "error": error, "box_task": report})
    return {"passed": all(row["passed"] for row in results),
            "task_success_count": sum(row["passed"] for row in results),
            "task_total_count": len(results), "worlds": results}


HOST_VALIDATION_POLICY = {
    "version": 1,
    "oracle": "validate_observations; unchanged per-world CubeEvidence and raw-clock/count checks",
    "parallelism": "persistent spawned CPU processes, capped by worlds/requested workers/affinity; one worker runs in the parent",
    "history_transport": "one float32 parent copy to shared memory per validation; child views are read-only",
    "timer": "validation includes history copy, dispatch, all oracle work and result collection; process setup and teardown separately reported",
}
_VALIDATION_CONTEXT = None
_VALIDATOR_ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                         "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                         "PXR_WORK_THREAD_LIMIT": "1"}


def _validation_initialize(name, shape, spec, phases, ready):
    global _VALIDATION_CONTEXT
    try:
        memory = shared_memory.SharedMemory(name=name)
        output = np.ndarray(shape, dtype=np.float32, buffer=memory.buf)
        output.flags.writeable = False
        _VALIDATION_CONTEXT = (memory, output, spec, phases)
        ready.put({"passed": True, "readonly": not output.flags.writeable,
                   "environment": {key: os.environ.get(key) for key in _VALIDATOR_ENVIRONMENT}})
    except Exception as error:
        ready.put({"passed": False, "error": str(error)})
        raise


def _validate_shared_world(work):
    world, clock_precision, step_count = work
    _, history, spec, phases = _VALIDATION_CONTEXT
    result = validate_observations(history[world:world+1], spec, phases,
                                   clock_precision=clock_precision, step_counts=[step_count])
    row = result["worlds"][0]
    row["world"] = world
    return row


class ObservationValidator:
    """Same serial oracle, independently applied to every immutable world history.

    Construct once outside simulation timing, reuse for warmups and measured
    episodes, and close in the case worker's finally block. validate() includes
    the only shared-history copy and all process dispatch/collection overhead.
    """
    def __init__(self, spec, phases, worlds, requested_workers):
        if type(worlds) is not int or worlds < 1 or type(requested_workers) is not int or requested_workers < 1:
            raise ValueError("Validation worlds and requested workers must be positive integers")
        if len(phases) != FRAMES or phases[-1] != "done":
            raise ValueError("Validation requires the complete canonical control phases")
        self.spec, self.phases = _base_spec(spec), np.asarray(phases)
        self.worlds = worlds
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        self.workers = min(worlds, requested_workers, affinity)
        self.shape = (worlds, FRAMES, OBSERVATION_SIZE)
        self.pool = self.memory = self.output = self.ready = None
        self.closed = False
        self.info = {"policy_version": HOST_VALIDATION_POLICY["version"],
                     "mode": "serial_parent" if self.workers == 1 else "persistent_spawn_pool",
                     "workers": self.workers, "requested_workers": requested_workers,
                     "affinity_limit": affinity, "shared_history_bytes": 0,
                     "child_history_readonly": None, "child_environment": None}
        if self.workers == 1:
            return
        required = int(np.prod(self.shape))*np.dtype(np.float32).itemsize
        if sys.platform.startswith("linux") and Path("/dev/shm").exists():
            stat = os.statvfs("/dev/shm")
            if required > stat.f_bavail*stat.f_frsize:
                raise ValueError("Insufficient shared memory for the independent validation history")
        context = mp.get_context("spawn")
        prior = {key: os.environ.get(key) for key in _VALIDATOR_ENVIRONMENT}
        module_path = str(Path(__file__).resolve().parent)
        added = module_path not in sys.path
        if added:
            sys.path.insert(0, module_path)
        try:
            self.memory = shared_memory.SharedMemory(create=True, size=required)
            self.output = np.ndarray(self.shape, dtype=np.float32, buffer=self.memory.buf)
            self.ready = context.Queue()
            os.environ.update(_VALIDATOR_ENVIRONMENT)
            self.pool = context.Pool(self.workers, initializer=_validation_initialize,
                initargs=(self.memory.name, self.shape, self.spec, self.phases, self.ready))
            receipts = [self.ready.get(timeout=120) for _ in range(self.workers)]
            if not all(item.get("passed") and item.get("readonly")
                       and item.get("environment") == _VALIDATOR_ENVIRONMENT for item in receipts):
                raise RuntimeError(f"Validation worker initialization failed: {receipts}")
            self.info.update(shared_history_bytes=required, child_history_readonly=True,
                             child_environment=dict(_VALIDATOR_ENVIRONMENT))
        except BaseException:
            self.close()
            raise
        finally:
            if added:
                sys.path.remove(module_path)
            for key, value in prior.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def validate(self, history, *, clock_precision, step_counts):
        if self.closed:
            raise RuntimeError("Validation workers have already been closed")
        history = np.asarray(history)
        if history.shape != self.shape or history.dtype != np.float32:
            raise ValueError("Validation requires the complete measured float32 history without conversion")
        if len(step_counts) != self.worlds:
            raise ValueError("An independent integration count is required for every world")
        if self.pool is None:
            return validate_observations(history, self.spec, self.phases,
                                         clock_precision=clock_precision, step_counts=step_counts)
        np.copyto(self.output, history, casting="no")
        # map preserves input order; each child evaluates its own world and the
        # original serial oracle is also retained for exact regression checks.
        rows = self.pool.map(_validate_shared_world,
            [(world, clock_precision, step_counts[world]) for world in range(self.worlds)],
            chunksize=max(1, self.worlds // (4*self.workers)))
        return {"passed": all(row["passed"] for row in rows),
                "task_success_count": sum(row["passed"] for row in rows),
                "task_total_count": len(rows), "worlds": rows}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.pool is not None:
                self.pool.terminate()
                self.pool.join()
        finally:
            self.pool = None
            self.output = None
            try:
                if self.memory is not None:
                    self.memory.close()
                    self.memory.unlink()
            finally:
                self.memory = None
                if self.ready is not None:
                    self.ready.close()
                    self.ready.join_thread()
                    self.ready = None
