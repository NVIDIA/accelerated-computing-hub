#!/usr/bin/env python3
"""Pinned ALOHA control-replay diagnostic: native MuJoCo vs MuJoCo Warp.

This is an adapted community diagnostic, not an official product benchmark,
not Newton API timing, and not validation of the blogs' two-cube task.
One invocation runs one backend/batch and saves every measured outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time
import traceback

import mujoco
import mujoco.rollout
import numpy as np
import warp as wp

PINNED = {
    "blog2": {"mujoco": "3.8.0", "mujoco-warp": "3.8.0.3", "warp-lang": "1.15.0"},
    "blog3": {"mujoco": "3.12.0", "mujoco-warp": "3.12.0", "warp-lang": "1.17.0", "newton": "1.6.0"},
}
MODEL_FIELDS = (
    "qpos0", "body_mass", "body_inertia", "body_pos", "body_quat",
    "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range",
    "geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size",
    "geom_friction", "geom_solref", "geom_solimp", "dof_damping", "dof_armature",
    "actuator_trnid", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm",
)
OPTION_FIELDS = ("timestep", "tolerance", "ls_tolerance", "iterations", "ls_iterations",
                 "impratio", "solver", "integrator", "cone", "disableflags", "enableflags")


@wp.kernel(enable_backward=False)
def targets(tape: wp.array2d(dtype=float), cursor: wp.array(dtype=int), ctrl: wp.array2d(dtype=float)):
    world, actuator = wp.tid()
    ctrl[world, actuator] = tape[cursor[0], actuator]


@wp.kernel(enable_backward=False)
def record(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
           act: wp.array2d(dtype=float), clock: wp.array(dtype=float),
           cursor: wp.array(dtype=int), output: wp.array3d(dtype=float),
           nacon: wp.array(dtype=int), nefc: wp.array(dtype=int),
           ncollision: wp.array(dtype=int), peaks: wp.array(dtype=int),
           nq: int, nv: int, na: int):
    world, field = wp.tid()
    value = float(0.0)
    if field == 0:
        value = clock[world]
        wp.atomic_max(peaks, 1, nefc[world])
        if world == 0:
            wp.atomic_max(peaks, 0, nacon[0])
            wp.atomic_max(peaks, 2, ncollision[0])
    elif field < 1 + nq:
        value = qpos[world, field - 1]
    elif field < 1 + nq + nv:
        value = qvel[world, field - 1 - nq]
    else:
        value = act[world, field - 1 - nq - nv]
    output[world, cursor[0], field] = value


@wp.kernel(enable_backward=False)
def advance(cursor: wp.array(dtype=int)):
    cursor[0] += 1


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def native_model_identity(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    fields = {}
    for name in MODEL_FIELDS:
        a = np.asarray(getattr(model, name))
        fields[name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                        "native_sha256": sha(a.tobytes()),
                        "float32_sha256": sha(a.astype(np.float32).tobytes()) if a.dtype.kind == "f" else None}
    return {"compiled_model_sha256": sha(buffer.tobytes()), "fields": fields,
            "options": {key: float(getattr(model.opt, key)) for key in OPTION_FIELDS}}


def upload_audit(native, uploaded):
    """Compare relevant effective arrays to the compiled source after quantization."""
    result = {"fields": {}, "options": {}, "passed": True}
    for key in MODEL_FIELDS:
        if not hasattr(uploaded, key):
            # qpos0 is host reset metadata rather than an uploaded dynamics field.
            result["fields"][key] = {"status": "not_exposed_by_backend"}
            continue
        actual = getattr(uploaded, key)
        actual = actual.numpy() if hasattr(actual, "numpy") else np.asarray(actual)
        expected = np.asarray(getattr(native, key))
        while actual.ndim > expected.ndim and actual.shape[0] == 1:
            actual = actual[0]
        if actual.size == expected.size:
            actual = actual.reshape(expected.shape)
        same = actual.shape == expected.shape and np.array_equal(actual, expected.astype(actual.dtype))
        result["fields"][key] = {"passed": bool(same), "actual_shape": list(actual.shape),
            "actual_dtype": str(actual.dtype), "actual_sha256": sha(actual.tobytes()),
            "maximum_native_difference": safe_float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape and actual.size else None}
        result["passed"] &= bool(same)
    for key in OPTION_FIELDS:
        target = "impratio_invsqrt" if key == "impratio" else key
        actual = getattr(uploaded.opt, target)
        actual = actual.numpy() if hasattr(actual, "numpy") else np.asarray(actual)
        actual = np.asarray(actual)
        expected = np.asarray(getattr(native.opt, key))
        if key == "impratio":
            expected = 1.0 / np.sqrt(np.maximum(expected, mujoco.mjMINVAL))
        expected = expected.astype(actual.dtype)
        same = bool(np.all(actual == expected))
        result["options"][key] = {"passed": same, "actual": actual.tolist(),
                                  "native": float(getattr(native.opt, key)), "uploaded_field": target}
        result["passed"] &= same
    return result


def load(args):
    path = args.assets / "scene_pot.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    with np.load(args.assets / "lift_pot.npz") as z:
        # Equal precision inputs; native dynamics remains float64.
        data.qpos[:] = z["qpos"][0].astype(np.float32).astype(np.float64)
        data.qvel[:] = z["qvel"][0].astype(np.float32).astype(np.float64)
        tape = z["ctrl"].astype(np.float32)
        times = z["times"].copy()
    dt = float(model.opt.timestep)
    decimation = max(1, round(float(times[1] - times[0]) / dt))
    if not np.isclose(dt * decimation, times[1] - times[0]):
        raise ValueError("Control and simulation timesteps must align exactly")
    tape = np.repeat(tape, decimation, axis=0)
    if args.steps:
        if args.steps > len(tape):
            raise ValueError("Do not cycle or extend the reference trajectory")
        tape = tape[:args.steps]
    if args.mode == "constant-control":
        tape = np.repeat(tape[:1], len(tape), axis=0)
    # Native tolerance defaults to 1e-8 while MJWarp clamps to 1e-6.
    # Explicit shared settings avoid claiming equality while running unequal tolerances.
    model.opt.tolerance = 1e-6
    model.opt.ls_tolerance = 1e-6
    data.ctrl[:] = tape[0]
    mujoco.mj_forward(model, data)
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return model, data, state, tape


class Native:
    def __init__(self, model, data, initial, tape, args):
        self.model = model
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        self.workers = min(args.workers, args.worlds, affinity)
        self.pool = mujoco.rollout.Rollout(nthread=self.workers)
        self.data = [mujoco.MjData(model) for _ in range(self.workers)]
        self.initial = np.repeat(initial[None], args.worlds, axis=0)
        self.control = np.ascontiguousarray(np.broadcast_to(tape.astype(np.float64), (args.worlds, *tape.shape)))
        self.output = np.empty((args.worlds, len(tape), len(initial)))
        self.sensors = np.empty((args.worlds, len(tape), model.nsensordata))
        self.nstep = len(tape)
        self.metadata = {"cpu_threads": self.workers, "native_path": "mujoco.rollout persistent C++ thread pool",
                         "timed_scope": "native reset, control replay, integration, full physics state history, CPU dispatch/completion",
                         "dtype": "float64", "capacity_checks": "native dynamic allocation; divergence clocks checked for every world and step",
                         "effective_model": native_model_identity(model)}

    def episode(self):
        start = time.perf_counter()
        self.pool.rollout([self.model] * len(self.initial), self.data, self.initial,
                          self.control, nstep=self.nstep, state=self.output,
                          sensordata=self.sensors, skip_checks=True)
        elapsed = time.perf_counter() - start
        return self.output, elapsed, 0.0, {}

    def close(self):
        self.pool.close()


class CUDA:
    def __init__(self, model, data, initial, tape, args):
        import mujoco_warp as mjw
        self.mjw = mjw
        wp.init()
        self.device = wp.get_device(args.device)
        if not self.device.is_cuda:
            raise ValueError("An explicit CUDA device is required; no CPU substitution")
        self.worlds = args.worlds
        self.nstep = len(tape)
        self.model = model
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(model)
            self.upload_comparison = upload_audit(model, self.m)
            if not self.upload_comparison["passed"]:
                write_json(args.output / "model-upload-mismatch.json", self.upload_comparison)
                raise ValueError("Effective GPU model differs from quantized native source; inspect model-upload-mismatch.json")
            self.d = mjw.put_data(model, data, nworld=args.worlds, nconmax=args.nconmax, njmax=args.njmax)
            self.tape = wp.array(tape, dtype=float)
            self.cursor = wp.zeros(1, dtype=int)
            self.peaks = wp.zeros(3, dtype=int)
            self.output = wp.empty((args.worlds, len(tape), len(initial)), dtype=float)
            self.qpos0 = wp.array(np.repeat(data.qpos.astype(np.float32)[None], args.worlds, axis=0))
            self.qvel0 = wp.array(np.repeat(data.qvel.astype(np.float32)[None], args.worlds, axis=0))
            self.ctrl0 = wp.array(np.repeat(tape[:1], args.worlds, axis=0))
            self.reset()
            self.step()
            self.reset()
            self.chunk = min(args.graph_steps, self.nstep)
            with wp.ScopedCapture() as capture:
                for _ in range(self.chunk):
                    self.step()
            self.graph = capture.graph
            self.reset()
            self.tail = None
            if self.nstep % self.chunk:
                with wp.ScopedCapture() as capture:
                    for _ in range(self.nstep % self.chunk):
                        self.step()
                self.tail = capture.graph
            self.reset()
        self.metadata = {"device": str(self.device), "device_name": self.device.name,
                         "timed_scope": "device reset, resident control replay, integration, full physics state history, launch and final synchronization",
                         "dtype": "float32", "graph_steps": self.chunk, "nconmax": args.nconmax,
                         "njmax": args.njmax, "naconmax": self.d.naconmax,
                         "ncollision_capacity": self.d.naconmax,
                         "actual_tolerance": self.m.opt.tolerance.numpy().tolist(),
                         "actual_ls_tolerance": self.m.opt.ls_tolerance.numpy().tolist(),
                         "effective_model": native_model_identity(model),
                         "upload_comparison": self.upload_comparison}

    def reset(self):
        self.mjw.reset_data(self.m, self.d)
        wp.copy(self.d.qpos, self.qpos0)
        wp.copy(self.d.qvel, self.qvel0)
        wp.copy(self.d.ctrl, self.ctrl0)
        self.cursor.zero_()
        self.peaks.zero_()

    def step(self):
        wp.launch(targets, (self.worlds, self.model.nu), inputs=[self.tape, self.cursor, self.d.ctrl])
        self.mjw.step(self.m, self.d)
        wp.launch(record, (self.worlds, self.output.shape[2]), inputs=[self.d.qpos, self.d.qvel,
            self.d.act, self.d.time, self.cursor, self.output, self.d.nacon, self.d.nefc,
            self.d.ncollision, self.peaks, self.model.nq, self.model.nv, self.model.na])
        wp.launch(advance, 1, inputs=[self.cursor])

    def episode(self):
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            start = time.perf_counter()
            self.reset()
            for _ in range(self.nstep // self.chunk):
                wp.capture_launch(self.graph)
            if self.tail is not None:
                wp.capture_launch(self.tail)
            wp.synchronize_device(self.device)
            elapsed = time.perf_counter() - start
            start = time.perf_counter()
            output = self.output.numpy()
            history_transfer = time.perf_counter() - start
            peaks = self.peaks.numpy().tolist()
            diag = {"peak_total_contacts": peaks[0], "peak_constraints_per_world": peaks[1],
                    "peak_broadphase_pairs": peaks[2], "integrations": int(self.cursor.numpy()[0])}
            if hasattr(self.d, "warning"):
                diag["warning"] = self.d.warning.numpy().tolist()
            if hasattr(self.d, "overflow"):
                diag["overflow"] = self.d.overflow.numpy().tolist()
            collection = time.perf_counter() - start
            diag["collection_timing"] = {"history_transfer_seconds": history_transfer,
                "diagnostic_collection_seconds": collection - history_transfer,
                "scope": "all history, capacity, clock/cursor and warning/overflow gathers; artifact I/O excluded"}
            return output, elapsed, collection, diag

    def close(self):
        pass


def physical_health(output, model, initial, require_lift):
    """Coarse scene sanity and the intended upstream pot/lid height assertions.

    The envelope is a declared coarse sanity guard derived from the authored
    scene extent and joint ranges. It is not a claim of accurate trajectories,
    solved grasp forces, or a complete pick/place task gate.
    """
    qpos = output[:, :, 1:1 + model.nq]
    qvel = output[:, :, 1 + model.nq:1 + model.nq + model.nv]
    checks, diagnostics = {}, {}
    quat_roundoff = float(np.sqrt(np.finfo(np.float32).eps))
    extent = float(model.stat.extent)
    if extent <= 0 or not np.isfinite(extent):
        raise ValueError("Expected a positive authored scene extent")
    policy = {"quaternion_norm_error_max": quat_roundoff,
              "quaternion_threshold_basis": "sqrt(float32 machine epsilon), numerical guard only",
              "free_translation_displacement_max_m": extent,
              "free_linear_speed_max_m_per_s": extent / model.opt.timestep,
              "bound_basis": "authored scene extent, complete joint ranges, and one physics timestep; coarse explosion guard"}
    quaternion_errors, free_excursions = [], []
    velocity_bound = np.full(model.nv, 2.0 * np.pi / model.opt.timestep)
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        q = int(model.jnt_qposadr[joint])
        v = int(model.jnt_dofadr[joint])
        if kind in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
            start = q + 3 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else q
            error = np.abs(np.linalg.norm(qpos[:, :, start:start + 4], axis=-1) - 1.0)
            quaternion_errors.append(float(error.max()))
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            excursion = np.linalg.norm(qpos[:, :, q:q + 3] - initial[1 + q:1 + q + 3], axis=-1)
            free_excursions.append(float(excursion.max()))
            velocity_bound[v:v + 3] = extent / model.opt.timestep
        elif bool(model.jnt_limited[joint]) and kind in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            lower, upper = model.jnt_range[joint]
            width = float(upper - lower)
            # Allow soft constraints their entire authored travel as a coarse
            # failure envelope; report actual excursions rather than implying
            # that soft joint limits must be satisfied exactly.
            checks[f"joint_{joint}_coarse_position_bound"] = bool(np.all((qpos[:, :, q] >= lower - width) & (qpos[:, :, q] <= upper + width)))
            velocity_bound[v] = width / model.opt.timestep
    checks["unit_free_ball_quaternions"] = all(x <= quat_roundoff for x in quaternion_errors)
    checks["free_translation_within_scene_extent"] = all(x <= extent for x in free_excursions)
    checks["coarse_velocity_bound"] = bool(np.all(np.abs(qvel) <= velocity_bound[None, None]))
    diagnostics.update({"max_quaternion_norm_error": safe_float(max(quaternion_errors, default=0.0)),
                        "max_free_translation_excursion_m": safe_float(max(free_excursions, default=0.0)),
                        "max_abs_qvel": safe_float(np.abs(qvel).max()),
                        "max_velocity_bound_fraction": safe_float(np.max(np.abs(qvel) / velocity_bound[None, None]))})
    # Verify names explicitly: the upstream regression test's pot label lacks
    # the trailing slash in this compiled scene, and an ID of -1 is invalid.
    pot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "partnet_100015/")
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "partnet_100015/link_0")
    if pot < 0 or lid < 0 or pot == lid:
        raise ValueError("Expected distinct named pot and lid bodies")
    frames = sorted(set([0, *[min(output.shape[1] - 1, max(0, round(t / model.opt.timestep) - 1)) for t in (.5, 1.5, 1.75)], output.shape[1] - 1]))
    data = mujoco.MjData(model)
    checkpoints = []
    for frame in frames:
        heights = []
        for world in range(output.shape[0]):
            data.qpos[:] = qpos[world, frame]
            mujoco.mj_kinematics(model, data)
            heights.append([float(data.xpos[pot, 2]), float(data.xpos[lid, 2])])
        heights = np.asarray(heights)
        checkpoints.append({"frame": frame, "raw_time_min": safe_float(output[:, frame, 0].min()),
                            "pot_z_min": safe_float(heights[:, 0].min()), "pot_z_max": safe_float(heights[:, 0].max()),
                            "lid_z_min": safe_float(heights[:, 1].min()), "lid_z_max": safe_float(heights[:, 1].max())})
        if frame == output.shape[1] - 1 and require_lift:
            checks["upstream_intended_pot_height"] = bool(np.all(heights[:, 0] > .069))
            checks["upstream_intended_lid_height"] = bool(np.all(heights[:, 1] > .16))
    diagnostics["checkpoints"] = checkpoints
    diagnostics["pot_body_id"] = pot
    diagnostics["lid_body_id"] = lid
    policy["upstream_coarse_lift"] = {"applied": require_lift, "pot_z_gt": .069, "lid_z_gt": .16,
        "source": "https://github.com/google-deepmind/mujoco_warp/blob/v3.12.0/mujoco_warp/_src/unroll_test.py#L38-L56",
        "adaptation": "benchmark asset scene and verified body IDs; not identical upstream fixture; no grasp/contact/hold success claim"}
    return {"checks": checks, "diagnostics": diagnostics, "policy": policy}


def validate(output, dt, backend, diag, metadata, model=None, initial=None, require_lift=False):
    expected = np.cumsum(np.full(output.shape[1], dt, dtype=np.float32 if backend == "gpu" else np.float64))
    finite = bool(np.isfinite(output).all())
    clocks = bool(np.allclose(output[:, :, 0], expected[None], atol=1e-6, rtol=1e-6))
    checks = {"finite_history": finite, "all_world_step_clocks": clocks}
    if backend == "gpu":
        checks["integration_count"] = diag["integrations"] == output.shape[1]
        checks["contacts_within_allocation"] = diag["peak_total_contacts"] <= metadata["naconmax"]
        checks["constraints_within_allocation"] = diag["peak_constraints_per_world"] <= metadata["njmax"]
        if metadata["ncollision_capacity"] is not None:
            checks["broadphase_within_allocation"] = diag["peak_broadphase_pairs"] <= metadata["ncollision_capacity"]
        if "overflow" in diag:
            # Upstream 3.12 combines capacity errors and convergence-limit
            # diagnostics. Retain every bit; exclude all capacity/error bits.
            checks["no_capacity_overflow_flags"] = not np.any(np.asarray(diag["overflow"], dtype=np.int64) & ~1536)
    health = physical_health(output, model, initial, require_lift) if finite and model is not None else None
    if health is not None:
        checks.update(health["checks"])
    return {"passed": all(checks.values()), "checks": checks, "physical_health": health,
            "final_raw_clock_min": safe_float(output[:, -1, 0].min()),
            "final_raw_clock_max": safe_float(output[:, -1, 0].max()),
            "scope": "coarse physical/numerical health plus source-derived final heights for full replay; no box-task claim"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=Path(__file__).parent / "aloha-pot")
    p.add_argument("--stack", choices=PINNED, required=True)
    p.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    p.add_argument("--worlds", type=int, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--mode", choices=["replay", "constant-control"], default="replay")
    p.add_argument("--steps", type=int, default=0, help="0 uses complete 1001-step upstream trajectory; nonzero is diagnostic prefix")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--graph-steps", type=int, default=10)
    p.add_argument("--nconmax", type=int, default=256)
    p.add_argument("--njmax", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.worlds, args.workers, args.repeats, args.warmups, args.graph_steps) < 1:
        p.error("Positive worlds/workers/repeats/warmups/graph steps required")
    if args.steps < 0:
        p.error("steps must be nonnegative")
    if args.output.exists():
        p.error("Output exists: preserve previous evidence and use a new directory")
    versions = {key: importlib.metadata.version(key) for key in PINNED[args.stack]}
    if versions != PINNED[args.stack]:
        raise RuntimeError(f"Pinned stack mismatch: {versions!r}")
    args.output.mkdir(parents=True)
    model, data, initial, tape = load(args)
    report = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "versions": versions, "platform": platform.platform(), "machine": platform.machine(),
              "source_sha256": sha(Path(__file__).read_bytes()), "scene_sha256": sha((args.assets / "scene_pot.xml").read_bytes()),
              "replay_sha256": sha((args.assets / "lift_pot.npz").read_bytes()), "control_tape_sha256": sha(tape.tobytes()),
              "steps": len(tape), "dt": float(model.opt.timestep),
              "model_dimensions": {key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "na", "nbody", "ngeom")},
              "solver_options": {key: float(getattr(model.opt, key)) for key in ("timestep", "tolerance", "ls_tolerance", "iterations", "ls_iterations", "impratio", "solver", "integrator", "cone")},
              "samples": [], "warmups": [], "status": "initializing"}
    write_json(args.output / "results.json", report)
    start = time.perf_counter()
    runner = None
    try:
        runner = (Native if args.backend == "cpu" else CUDA)(model, data, initial, tape, args)
        report["setup_seconds"] = time.perf_counter() - start
        report["backend_metadata"] = runner.metadata
        for index in range(args.warmups + args.repeats):
            output, elapsed, collection, diag = runner.episode()
            start = time.perf_counter()
            verdict = validate(output, float(model.opt.timestep), args.backend, diag, runner.metadata,
                               model, initial, args.mode == "replay" and len(tape) == 1001)
            validation_seconds = time.perf_counter() - start
            sample = {"simulation_seconds": elapsed, "output_collection_seconds": collection,
                      "history_transfer_seconds": diag.get("collection_timing", {}).get("history_transfer_seconds", collection),
                      "validation_seconds": validation_seconds, "diagnostics": diag, "validation": verdict}
            label = "warmup" if index < args.warmups else "measured"
            np.savez_compressed(args.output / f"{label}-{index:02d}-endpoints.npz",
                                first_state=output[:, 0], final_state=output[:, -1])
            report["warmups" if index < args.warmups else "samples"].append(sample)
            report["status"] = "running" if verdict["passed"] else "numerical_check_failed"
            write_json(args.output / "results.json", report)
            if not verdict["passed"]:
                break
        valid = len(report["samples"]) == args.repeats and all(s["validation"]["passed"] for s in report["samples"] + report["warmups"])
        report["status"] = "complete" if valid else "numerical_check_failed"
        report["median_simulation_seconds"] = statistics.median(s["simulation_seconds"] for s in report["samples"]) if valid else None
        report["median_checked_history_seconds"] = statistics.median(s["simulation_seconds"] + s["output_collection_seconds"] + s["validation_seconds"] for s in report["samples"]) if valid else None
        report["world_steps_per_second"] = args.worlds * len(tape) / report["median_simulation_seconds"] if valid else None
        write_json(args.output / "results.json", report)
        print(json.dumps({key: report[key] for key in ("status", "median_simulation_seconds", "world_steps_per_second")}))
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                           "phase": "constructor" if runner is None else "episode_or_validation"}
        report["median_simulation_seconds"] = None
        report["median_checked_history_seconds"] = None
        report["world_steps_per_second"] = None
        write_json(args.output / "results.json", report)
        raise
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
