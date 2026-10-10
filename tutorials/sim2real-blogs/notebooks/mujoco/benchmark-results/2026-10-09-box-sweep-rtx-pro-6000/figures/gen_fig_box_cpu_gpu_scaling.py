#!/usr/bin/env python3
"""Render two-cube box scaling from finished raw JSON or JSON.gz reports.

Private staging usage (no physics packages are imported):
  python gen_fig_box_cpu_gpu_scaling.py --article 2 --validate-only
  python gen_fig_box_cpu_gpu_scaling.py --article 2 --metric both

When copied to a package's figures/ directory, defaults to that package's
{so101,rebot}/results.json.gz (falling back to results.json). In private staging,
--article N defaults to this script's adjacent articleN/ report directory.
--package-dir and --output-dir override these locations.

Requires only Python stdlib for auditing; rendering optionally imports matplotlib
(validated renderer environment: matplotlib 3.10.7). Never install plotting tools
into, import, or modify the pinned physics environment.

Execution revision v3 requires the signed host-validation policy and actual
validator allocation. The physical acceptance protocol remains version 2.

Only the full canonical two-cube receiving-box protocol is accepted: 40 simulated
seconds, 2000 control frames, 40000 physics steps of 0.001 s, protocol version 2,
one passing warm-up and five
passing measured repetitions, 61 float32 physical observations per world/frame.
Every world must retain both cubes' grasp/lift/carry/release/settling evidence.
Historical stacking measurements are rejected. Failed cases retain no timings.

The default renders two complementary two-panel figures, one panel per robot:
* simulation: reset, commands, steps, forward evaluations and state/contact
  recording, with CPU dispatch/completion or CUDA synchronization;
* checked: simulation + final host collection/download + validation, summed for
  each repetition BEFORE computing median/min/max. This is not total process
  time: model/tape preparation, compilation, setup and warm-up remain excluded.
Whiskers are the five-sample observed min/max, never confidence intervals.
PNG is 300 dpi; PDF/SVG are vector; SVG whitespace is normalized. Exact accepted
samples, component timings, exclusions and source hashes accompany the figures.
No figures or plotted-data files are written when --validate-only is requested.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import tempfile

COUNTS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
ROBOTS = {"so101": "SO-101", "rebot": "reBot"}
COLORS = {"cpu": "#0072B2", "gpu": "#D55E00"}  # Okabe–Ito
BACKENDS = {2: {"cpu": "mujoco", "gpu": "mjwarp"},
            3: {"cpu": "newton_cpu", "gpu": "newton_cuda"}}
VERSIONS = {2: {"mujoco": "3.8.0", "mujoco-warp": "3.8.0.3", "warp-lang": "1.15.0"},
            3: {"mujoco": "3.12.0", "mujoco-warp": "3.12.0", "warp-lang": "1.17.0", "newton": "1.6.0"}}
STEMS = {"simulation": "fig_box_cpu_gpu_scaling", "checked": "fig_box_cpu_gpu_checked_results"}
EXPECTED_ACCEPTANCE = {
    "protocol_version": 2,
    "clock_validation": "every raw float32 observation matches sequential backend-precision integration within one observation ULP; independent per-world integration count",
    "clock_observation_ulp_allowance": 1,
    "task": "two_cube_pick_place_into_box", "frames": 2000, "physics_steps": 40000,
    "simulated_seconds": 40.0, "control_hz": 50,
    "bilateral_close_frames": 10, "minimum_jaw_normal_force_N": 1e-5,
    "whole_cube_lift_m": .01, "consecutive_loaded_lift_frames": 25,
    "loaded_carry_frames": 10, "horizontal_carry_m": .05,
    "whole_cube_box_tolerance_m": .003, "release_open_fraction": .8,
    "detached_settled_frames": 50, "maximum_settled_corner_speed_m_s": .04,
    "final_jaw_clearance_m": .02, "ordered_cube_cycles": ["red_cube", "blue_cube"],
    "observation_fields": 61, "observation_dtype": "float32",
}
HOST_VALIDATION_POLICY = {
    "version": 1,
    "oracle": "validate_observations; unchanged per-world CubeEvidence and raw-clock/count checks",
    "parallelism": "persistent spawned CPU processes, capped by worlds/requested workers/affinity; one worker runs in the parent",
    "history_transport": "one float32 parent copy to shared memory per validation; child views are read-only",
    "timer": "validation includes history copy, dispatch, all oracle work and result collection; process setup and teardown separately reported",
}
VALIDATOR_ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                         "MKL_NUM_THREADS": "1", "PXR_WORK_THREAD_LIMIT": "1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_publication_designation(records, label):
    for record in records:
        require(record.get("diagnostic_only", False) is False
                and record.get("publication_eligible", True) is True,
                f"{label}: explicitly diagnostic-only or excluded from publication")


def finite(value, *, minimum=None, strict=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (minimum is None or (value > minimum if strict else value >= minimum)))


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def read_report(path):
    packed = path.read_bytes()
    raw = gzip.decompress(packed) if path.suffix == ".gz" else packed
    return json.loads(raw), {"file": f"{path.parent.name}/{path.name}",
                            "sha256": sha256(packed), "uncompressed_sha256": sha256(raw)}


def check_workload(workload, label, *, require_compiled_model=True, article=None):
    article = article or (3 if "physics_dt_seconds" in workload else 2)
    require(workload.get("task") == "two_cube_pick_place_into_box",
            f"{label}: expected the two-cube box task; historical stacking is excluded")
    require(workload.get("frames") == 2000 and workload.get("substeps") == 20
            and workload.get("physics_steps") == 40000 and workload.get("simulated_seconds") == 40,
            f"{label}: expected the complete 40-second / 40000-step task")
    require(workload.get("dt" if article == 2 else "physics_dt_seconds") == .001,
            f"{label}: expected protocol v2 integration timestep of 0.001 s")
    if article == 2:
        require(workload.get("protocol_version") == 2 and workload.get("control_dt") == .02,
                f"{label}: expected protocol version 2 and 50 Hz control")
    else:
        policy = workload.get("backend_policy", {})
        require(policy.get("collision_detection") == "backend defaults; no collider flag override"
                and policy.get("native_clock_precision") == "float64"
                and policy.get("gpu_clock_precision") == "float32",
                f"{label}: backend or clock policy differs")
    acceptance = workload.get("acceptance", {})
    require(all(acceptance.get(k) == v for k, v in EXPECTED_ACCEPTANCE.items()),
            f"{label}: acceptance or 61-field float32 recording protocol differs")
    require("61" in workload.get("output", "") and "float32" in workload.get("output", ""),
            f"{label}: missing common 61-field float32 recording policy")
    keys = ["comparison_signature", "control_tape_sha256", "initial_robot_ctrl_sha256"]
    # Newton attaches its compiled model identity only after successful setup.
    # An allocation/setup failure still belongs in the plot as an untimed gap.
    if require_compiled_model or workload.get("model_mjb_sha256") is not None:
        keys.append("model_mjb_sha256")
    for key in keys:
        require(isinstance(workload.get(key), str) and len(workload[key]) == 64
                and all(c in "0123456789abcdef" for c in workload[key]),
                f"{label}: missing {key}")
    initial = workload.get("initial_robot_ctrl", [])
    require(isinstance(initial, list) and initial and all(finite(v) for v in initial),
            f"{label}: missing initial robot pose")
    require(sha256(struct.pack(f"<{len(initial)}f", *initial)) == workload["initial_robot_ctrl_sha256"],
            f"{label}: initial-pose hash disagrees with recorded controls")
    excluded = {"comparison_signature"}
    if article == 3:
        # The engine appends these device-specific compiled-model fields after
        # hashing the shared protocol. They cannot enter the shared signature.
        excluded.update({"model_mjb_sha256", "initial_compiled_qpos0_sha256",
                         "initial_newton_joint_q_first_world_sha256",
                         "initial_newton_joint_qd_first_world_sha256",
                         "model_dimensions", "solver_options", "backend_settings"})
    protocol = {k: v for k, v in workload.items() if k not in excluded}
    require(sha256(json.dumps(protocol, sort_keys=True).encode()) == workload["comparison_signature"],
            f"{label}: stale or inconsistent comparison signature")
    if require_compiled_model:
        options = workload.get("model_options" if article == 2 else "solver_options", {})
        require(all(options.get(k) == v for k, v in
                    {"timestep": .001, "iterations": 100, "ls_iterations": 50, "impratio": 100}.items()),
                f"{label}: different actual solver settings")


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float32_ulp(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0] - value


@lru_cache(maxsize=2)
def expected_backend_time(precision):
    require(precision in ("float32", "float64"), "Unknown backend clock precision")
    cast = float32 if precision == "float32" else float
    current, timestep = cast(0), cast(.001)
    for _ in range(40000):
        current = cast(current + timestep)
    return current


def check_clock(world, label, precision, step_count):
    clock = world.get("clock", {})
    expected = float32(expected_backend_time(precision))
    require(clock.get("passed") is True and clock.get("integration_count_passed") is True,
            f"{label}: missing or failed raw-clock/integration proof")
    require(clock.get("backend_clock_precision") == precision
            and clock.get("recorded_clock_precision") == "float32"
            and type(clock.get("allowed_observation_ulps")) is int and clock["allowed_observation_ulps"] == 1,
            f"{label}: clock precision or ULP allowance differs")
    require(type(step_count) is int and step_count == 40000
            and type(clock.get("recorded_physics_steps")) is int and clock["recorded_physics_steps"] == step_count
            and type(clock.get("expected_physics_steps")) is int and clock["expected_physics_steps"] == 40000,
            f"{label}: raw-clock receipt disagrees with actual integration count")
    require(type(clock.get("mismatched_frame_count")) is int and clock["mismatched_frame_count"] == 0
            and finite(clock.get("maximum_observation_ulp_error"), minimum=0)
            and clock["maximum_observation_ulp_error"] <= 1,
            f"{label}: a recorded frame clock missed its precision-aware expectation")
    require(clock.get("expected_final_raw_time_s") == expected
            and clock.get("nominal_integrated_duration_s") == 40,
            f"{label}: clock expectation differs from sequential integration")
    final = world.get("final_time_s")
    require(finite(final) and abs(final - expected) <= float32_ulp(expected),
            f"{label}: raw final clock differs from the actual backend expectation")
    require(abs(final - expected) / float32_ulp(expected) <= clock["maximum_observation_ulp_error"],
            f"{label}: final raw clock contradicts its maximum-error receipt")


def check_backend_settings(case, article, series, label):
    if article == 2:
        settings = case.get("backend_settings")
        require(settings == case.get("device_info", {}).get("backend_options"),
                f"{label}: duplicated native/uploaded backend metadata disagree")
    else:
        settings = case.get("workload", {}).get("backend_settings")
    require(isinstance(settings, dict), f"{label}: missing actual backend settings")
    expected_backend = "native_mujoco" if series == "cpu" else "mujoco_warp"
    expected_precision = "float64" if series == "cpu" else "float32"
    if article == 3 or "active_backend" in settings or "active_clock_precision" in settings:
        require(settings.get("active_backend") == expected_backend
                and settings.get("active_clock_precision") == expected_precision,
                f"{label}: active backend or clock precision differs")
    require(settings.get("collision_policy") == "backend defaults; no collider flag override",
            f"{label}: collision flags were overridden")
    native = settings.get("native_requested", {})
    require(isinstance(native, dict), f"{label}: missing native options")
    required_native = ("timestep", "tolerance", "ls_tolerance", "ccd_tolerance", "impratio",
                       "iterations", "ls_iterations", "ccd_iterations", "solver", "integrator",
                       "cone", "disableflags", "enableflags")
    require(all(finite(native.get(k), minimum=0) for k in required_native),
            f"{label}: incomplete or nonfinite native option snapshot")
    require(all(native.get(k) == v for k, v in {"timestep": .001, "iterations": 100,
            "ls_iterations": 50, "impratio": 100, "solver": 2, "integrator": 3, "cone": 1,
            "disableflags": 0 if article == 2 else 524288, "enableflags": 0}.items()),
            f"{label}: native options differ from the pinned backend defaults")
    source_options = case["workload"]["model_options" if article == 2 else "solver_options"]
    require(all(source_options.get(k) == native[k] for k in source_options),
            f"{label}: native settings disagree with compiled model metadata")
    uploaded = settings.get("uploaded_gpu_effective")
    if series == "cpu":
        require(uploaded is None, f"{label}: CPU unexpectedly declares uploaded GPU options")
        return settings
    require(isinstance(uploaded, dict), f"{label}: GPU effective options are missing")
    # Pin-specific IO policy: clamp tolerance at 1e-6, cast other numerical
    # options to float32, and upload inverse sqrt(impratio). Keep actual values.
    expected_arrays = {"timestep": float32(native["timestep"]),
                       "tolerance": float32(max(native["tolerance"], 1e-6)),
                       "ls_tolerance": float32(native["ls_tolerance"]),
                       "ccd_tolerance": float32(native["ccd_tolerance"]),
                       "impratio_invsqrt": float32(1 / math.sqrt(native["impratio"]))}
    worlds = case["config"]["worlds"]
    for name, expected in expected_arrays.items():
        item = uploaded.get(name, {})
        size = 1 if article == 2 or name == "timestep" else worlds
        require(isinstance(item, dict) and item.get("shape") == [size] and item.get("dtype") == "float32",
                f"{label}: wrong effective {name} array shape/dtype")
        require(("uniform_value" in item) != ("values" in item),
                f"{label}: ambiguous effective {name} values")
        values = [item["uniform_value"]] if "uniform_value" in item else item["values"]
        require(isinstance(values, list) and len(values) == (1 if "uniform_value" in item else size)
                and all(finite(v) and v == expected for v in values),
                f"{label}: unexpected uploaded effective {name}")
    for name in ("iterations", "ls_iterations", "ccd_iterations", "solver", "integrator", "cone",
                 "disableflags", "enableflags"):
        require(type(uploaded.get(name)) is int and uploaded[name] == native[name],
                f"{label}: effective {name} differs from the native request")
    return settings


def check_world(world, label, clock_precision, step_count):
    require(all(world.get(k) is True for k in ("passed", "finite", "time_progression")),
            f"{label}: failed world/time/finite evidence")
    check_clock(world, label, clock_precision, step_count)
    task = world.get("box_task", {})
    require(task.get("task") == "two_cube_pick_place_into_box" and task.get("success") is True
            and task.get("phase") == "done" and task.get("frames") == 2000
            and task.get("simulation_seconds") == 40 and finite(task.get("tool_clearance_m"), minimum=.02),
            f"{label}: incomplete two-cube box task or final withdrawal")
    objects = task.get("objects", {})
    require(set(objects) == {"red_cube", "blue_cube"}, f"{label}: both cubes are required")
    lower, upper = task.get("bin_lower", []), task.get("bin_upper", [])
    require(len(lower) == len(upper) == 3 and all(finite(v) for v in lower + upper)
            and all(a < b for a, b in zip(lower, upper)), f"{label}: invalid receiving-box bounds")
    for name, cube in objects.items():
        prefix = f"{label}/{name}"
        require(all(cube.get(k) is True for k in (
            "success", "initially_outside_bin", "grasped", "lifted", "carried", "release_commanded", "released", "inside"))
            and cube.get("dropped_before_release") is False, f"{prefix}: incomplete physical evidence")
        for key, minimum in (("max_loaded_bilateral_frames", 10), ("full_lift_frames", 25),
                             ("carry_frames", 10), ("detached_settled_frames", 50)):
            require(type(cube.get(key)) is int and cube[key] >= minimum, f"{prefix}: invalid {key}")
        for key, minimum in (("min_lift_clearance_m", .01), ("carry_distance_m", .05), ("release_open_fraction", .8)):
            require(finite(cube.get(key), minimum=minimum), f"{prefix}: invalid {key}")
        require(finite(cube.get("min_jaw_normal_force_N"), minimum=1e-5, strict=True),
                f"{prefix}: no loaded bilateral solved-contact evidence")
        require(finite(cube.get("max_point_speed"), minimum=0) and cube["max_point_speed"] < .04
                and cube.get("jaw_contacts") == [0, 0], f"{prefix}: not detached and settled")
        require(cube.get("frame") == 2000 and cube.get("time") == 40, f"{prefix}: incomplete final observation")
        lo, hi = cube.get("bounds_min", []), cube.get("bounds_max", [])
        require(len(lo) == len(hi) == 3 and all(finite(v) for v in lo + hi)
                and all(a <= b for a, b in zip(lo, hi))
                and all(a >= b - .003 for a, b in zip(lo, lower))
                and all(a <= b + .003 for a, b in zip(hi, upper)), f"{prefix}: whole-cube containment failed")
        times = [cube.get(k) for k in ("grasp_time", "lift_time", "carry_time", "release_command_time", "release_time")]
        require(all(finite(t, minimum=0, strict=True) and t <= 40 for t in times)
                and times == sorted(times), f"{prefix}: invalid grasp-to-release event ordering")
    require(objects["red_cube"]["release_time"] <= objects["blue_cube"]["grasp_time"],
            f"{label}: cubes were not handled sequentially")


def check_host_validation(case, label):
    config = case["config"]
    requested = config.get("validation_workers")
    policy = case["workload"].get("host_validation_policy", {})
    require(type(requested) is int and requested > 0
            and policy == {**HOST_VALIDATION_POLICY, "requested_workers": requested},
            f"{label}: missing or different v3 host-validation policy")
    info = case.get("host_validation", {})
    require(type(info.get("policy_version")) is int and info["policy_version"] == 1
            and info.get("requested_workers") == requested
            and type(info.get("affinity_limit")) is int and info["affinity_limit"] > 0,
            f"{label}: incomplete actual host-validation allocation")
    workers = min(config["worlds"], requested, info["affinity_limit"])
    require(type(info.get("workers")) is int and info["workers"] == workers,
            f"{label}: actual validation worker count differs from its recorded cap")
    if workers == 1:
        require(info.get("mode") == "serial_parent" and type(info.get("shared_history_bytes")) is int
                and info["shared_history_bytes"] == 0
                and info.get("child_history_readonly") is None and info.get("child_environment") is None,
                f"{label}: inconsistent single-worker validation metadata")
    else:
        require(info.get("mode") == "persistent_spawn_pool"
                and type(info.get("shared_history_bytes")) is int
                and info["shared_history_bytes"] == config["worlds"] * 2000 * 61 * 4
                and info.get("child_history_readonly") is True
                and info.get("child_environment") == VALIDATOR_ENVIRONMENT,
                f"{label}: missing persistent pool, immutable history or isolated worker evidence")
    timings = case.get("timings", {})
    components = ("preparation_seconds", "backend_setup_seconds", "validation_setup_seconds",
                  "validation_teardown_seconds", "case_seconds")
    require(all(finite(timings.get(k), minimum=0) for k in components),
            f"{label}: validator setup/teardown or case timing is missing")
    episode_cost = sum(sample[k] for sample in case["warmup_samples"] + case["samples"]
                       for k in ("simulation_seconds", "output_transfer_seconds", "validation_seconds"))
    minimum_case_cost = episode_cost + sum(timings[k] for k in components if k != "case_seconds")
    require(timings["case_seconds"] >= minimum_case_cost
            or math.isclose(timings["case_seconds"], minimum_case_cost, rel_tol=1e-9, abs_tol=1e-9),
            f"{label}: recorded costs overlap or exceed complete case wall time")
    return {"policy": policy, "actual": info,
            "setup_seconds": timings["validation_setup_seconds"],
            "teardown_seconds": timings["validation_teardown_seconds"],
            "case_seconds": timings["case_seconds"]}


def check_episode(sample, case, label):
    worlds = case["config"]["worlds"]
    validation = sample.get("validation", {})
    checks = validation.get("worlds", [])
    diagnostics = sample.get("diagnostics", {})
    backend = case["config"]["backend"]
    if backend == "newton_cuda":
        require("peak_broadphase_pairs_all_worlds" in diagnostics
                and "naconmax_all_worlds" in diagnostics,
                f"{label}: missing raw broadphase high-water evidence; unguarded v2 smoke is diagnostic only")
    if backend in ("mjwarp", "newton_cuda"):
        mandatory = {"peak_contacts_all_worlds", "peak_constraints_per_world", "peak_broadphase_pairs_all_worlds",
                     "overflow_flags_per_world", "capacity_flags_per_world", "iteration_warning_flags_per_world",
                     "nonfinite_state_per_world"}
        require(mandatory <= diagnostics.keys(), f"{label}: GPU raw/sticky/nonfinite guard evidence is incomplete")
        require(isinstance(diagnostics["nonfinite_state_per_world"], list)
                and len(diagnostics["nonfinite_state_per_world"]) == worlds
                and all(type(v) is int and v == 0 for v in diagnostics["nonfinite_state_per_world"]),
                f"{label}: missing or failed per-world finite-state guard")
        flag_keys = ("overflow_flags_per_world", "capacity_flags_per_world", "iteration_warning_flags_per_world")
        unavailable = all(diagnostics[k] is None for k in flag_keys)
        require((backend == "mjwarp" and unavailable) or all(
                isinstance(diagnostics[k], list) and len(diagnostics[k]) == worlds
                and all(type(v) is int and v >= 0 for v in diagnostics[k]) for k in flag_keys),
                f"{label}: incomplete or unsupported sticky flag evidence")
        if not unavailable:
            # Pinned MuJoCo Warp 3.12: ITERATIONS=512, LS_ITERATIONS=1024.
            # All other bits are capacity failures, including unknown bits;
            # a report must not relabel a capacity bit as a harmless warning.
            iteration_mask = 512 | 1024
            require(all(iteration == (total & iteration_mask)
                        and capacity == (total & ~iteration_mask)
                        for total, capacity, iteration in
                        zip(*(diagnostics[k] for k in flag_keys))),
                    f"{label}: sticky overflow and classified flags disagree")
        require(all(type(diagnostics[k]) is int and diagnostics[k] >= 0 for k in
                    ("peak_contacts_all_worlds", "peak_broadphase_pairs_all_worlds"))
                and isinstance(diagnostics["peak_constraints_per_world"], list)
                and all(type(v) is int and v >= 0 for v in diagnostics["peak_constraints_per_world"]),
                f"{label}: invalid raw capacity high-water counters")
    require(sample.get("passed") is True and validation.get("passed") is True
            and sample.get("task_success_count") == sample.get("task_total_count") == worlds
            and validation.get("task_success_count") == validation.get("task_total_count") == worlds
            and len(checks) == worlds and {w.get("world") for w in checks} == set(range(worlds)),
            f"{label}: missing or failed environment evidence")
    require(finite(sample.get("simulation_seconds"), minimum=0, strict=True)
            and all(finite(sample.get(k), minimum=0) for k in ("output_transfer_seconds", "validation_seconds")),
            f"{label}: invalid component times")
    require(diagnostics.get("capacity_passed") is True, f"{label}: capacity check failed")
    for key in ("capacity_flags_per_world", "nonfinite_state_per_world"):
        if diagnostics.get(key) is not None:
            require(len(diagnostics[key]) == worlds and not any(diagnostics[key]), f"{label}: {key} failed")
    info = case.get("device_info", {})
    contact_bound = diagnostics.get("naconmax_all_worlds", info.get("naconmax"))
    constraint_bound = diagnostics.get("njmax_per_world", info.get("njmax"))
    for key in ("peak_contacts_all_worlds", "peak_broadphase_pairs_all_worlds"):
        if key in diagnostics:
            require(finite(contact_bound, minimum=0) and finite(diagnostics[key], minimum=0)
                    and diagnostics[key] <= contact_bound, f"{label}: {key} exceeds capacity")
    if "peak_constraints_per_world" in diagnostics:
        peaks = diagnostics["peak_constraints_per_world"]
        require(len(peaks) == worlds and finite(constraint_bound, minimum=0)
                and all(finite(v, minimum=0) and v <= constraint_bound for v in peaks),
                f"{label}: constraints exceed capacity")
    cpu = backend in ("mujoco", "newton_cpu")
    clock_precision = "float64" if cpu else "float32"
    if backend in ("mujoco", "mjwarp"):
        step_counts = diagnostics.get("integration_steps_per_world")
        require(diagnostics.get("clock_precision") == clock_precision, f"{label}: diagnostic clock precision differs")
    elif cpu:
        worker_counts = {w.get("world"): w.get("recorded_physics_steps") for w in diagnostics.get("worlds", [])}
        step_counts = [worker_counts.get(i) for i in range(worlds)]
    else:
        step_counts = diagnostics.get("recorded_physics_steps_per_world")
        require(diagnostics.get("recorded_control_frames") == 2000
                and diagnostics.get("frame_cursor_derived_physics_steps") == 40000,
                f"{label}: incomplete GPU frame cursor")
    require(isinstance(step_counts, list) and len(step_counts) == worlds
            and all(type(n) is int and n == 40000 for n in step_counts),
            f"{label}: each world needs an independent actual 40000-step integration count")
    if cpu:
        worker_checks = diagnostics.get("worlds", [])
        require(diagnostics.get("completed_worlds") == worlds and len(worker_checks) == worlds
                and {w.get("world") for w in worker_checks} == set(range(worlds)),
                f"{label}: CPU did not complete every environment")
        for check in worker_checks:
            require(check.get("capacity_passed") is True and not check.get("warnings")
                    and check.get("finite_all_bodies", check.get("finite_state")) is True
                    and check.get("recorded_physics_steps") == step_counts[check["world"]]
                    and finite(check.get("final_time_seconds"))
                    and abs(check["final_time_seconds"] - expected_backend_time("float64")) <= float32_ulp(40.),
                    f"{label}: incomplete CPU environment or clock")
    elif "recorded_physics_steps" in diagnostics:
        require(diagnostics["recorded_physics_steps"] == 40000, f"{label}: incomplete GPU frame-derived step count")
    for world in checks:
        check_world(world, f"{label}/world{world['world']}", clock_precision, step_counts[world["world"]])


def statistics_for(times):
    return {"samples_seconds": times, "median_seconds": statistics.median(times),
            "min_seconds": min(times), "max_seconds": max(times)}


def extract_case(case, robot, series, worlds, article, *, expected_repeats=5):
    # The publication path always uses five. The private packager can audit
    # actual diagnostic reports with their recorded repeat count, without
    # rendering them or changing any observations.
    config = case["config"]
    label = f"{robot}/{series}/{worlds}"
    require(config.get("task") == "box", f"{label}: historical stacking is excluded; expected task=box")
    require(config.get("scope") == ("replay" if article == 2 else "newton-batch"), f"{label}: wrong execution scope")
    require(config.get("robot") == robot and config.get("worlds") == worlds
            and config.get("backend") == BACKENDS[article][series], f"{label}: case identity differs")
    require(config.get("frames") == 2000 and config.get("substeps") == 20
            and config.get("repeats") == expected_repeats and config.get("warmups") == 1,
            f"{label}: expected 2000 frames, 20 substeps, one warm-up and {expected_repeats} repetitions")
    workload = case.get("workload", {})
    accepted = case.get("status") == "passed"
    if accepted:
        check_publication_designation((case, config, workload, case.get("summary", {})), label)
    require(case.get("summary", {}).get("eligible", False) is accepted,
            f"{label}: status and summary eligibility differ")
    if not accepted:
        require(all(case.get("summary", {}).get(key) is None for key in (
            "median_seconds", "min_seconds", "max_seconds", "median_with_transfer_and_validation_seconds",
            "successful_tasks_per_second", "world_steps_per_second", "aggregate_real_time_factor")),
            f"{label}: failed case exposes an accepted summary timing; raw sample times are diagnostic only")
    if workload:
        check_workload(workload, label, require_compiled_model=accepted, article=article)
    if accepted:
        require(workload, f"{label}: passing case lacks its box protocol")
    workers = case.get("device_info", {}).get("cpu_workers", case.get("device_info", {}).get("cpu_threads")) if series == "cpu" else None
    if accepted and series == "cpu":
        require(workers == min(32, worlds), f"{label}: CPU allocation differs from min(32, environments)")
    if accepted and series == "gpu":
        require(str(config.get("device", "")).startswith("cuda:"), f"{label}: expected one selected CUDA device")
    backend_settings = check_backend_settings(case, article, series, label) if accepted else None
    measured, warmups = case.get("samples", []), case.get("warmup_samples", [])
    point = {"robot": robot, "series": series, "backend": config["backend"], "worlds": worlds,
             "case_id": case.get("case_id", config.get("case_id")), "status": case.get("status"),
             "accepted": accepted, "cpu_workers": workers, "warmup_count": len(warmups),
             "measured_count": len(measured), "comparison_signature": workload.get("comparison_signature"),
             "model_sha256": workload.get("model_mjb_sha256"), "acceptance": workload.get("acceptance"),
             "backend_settings": backend_settings,
             "clock_evidence": None,
             "host_validation": None,
             "simulation": None, "checked": None, "components_seconds": None,
             "excluded_reason": None if accepted else case.get("error") or case.get("diagnostics", {}).get("error") or case.get("status"),
             "failed_world_samples": sum(sum(w.get("passed") is not True for w in s.get("validation", {}).get("worlds", [])) for s in warmups + measured)}
    if accepted:
        require(len(measured) == expected_repeats and len(warmups) == 1, f"{label}: incomplete repetition evidence")
        require([s.get("repeat") for s in measured] == list(range(expected_repeats)),
                f"{label}: measured repetitions are duplicated, missing or reordered")
        for phase, episodes in (("warmup", warmups), ("measured", measured)):
            for i, sample in enumerate(episodes):
                check_episode(sample, case, f"{label}/{phase}{i}")
        point["host_validation"] = check_host_validation(case, label)
        worlds_checked = [w for sample in warmups + measured for w in sample["validation"]["worlds"]]
        point["clock_evidence"] = {
            "world_episodes_checked": len(worlds_checked), "actual_steps_per_world": 40000,
            "backend_clock_precision": "float64" if series == "cpu" else "float32",
            "recorded_clock_precision": "float32", "allowed_observation_ulps": 1,
            "maximum_observation_ulp_error": max(w["clock"]["maximum_observation_ulp_error"] for w in worlds_checked),
            "expected_final_raw_time_s": worlds_checked[0]["clock"]["expected_final_raw_time_s"],
            "minimum_final_raw_time_s": min(w["final_time_s"] for w in worlds_checked),
            "maximum_final_raw_time_s": max(w["final_time_s"] for w in worlds_checked),
            "nominal_integrated_duration_s": 40.0}
        components = {k: [s[k] for s in measured] for k in ("simulation_seconds", "output_transfer_seconds", "validation_seconds")}
        point["components_seconds"] = components
        point["simulation"] = statistics_for(components["simulation_seconds"])
        point["checked"] = statistics_for([sum(s[k] for k in components) for s in measured])
        summary = case["summary"]
        for key in ("median_seconds", "min_seconds", "max_seconds"):
            require(math.isclose(point["simulation"][key], summary[key], rel_tol=1e-12), f"{label}: {key} disagrees with raw samples")
        require(math.isclose(point["checked"]["median_seconds"], summary["median_with_transfer_and_validation_seconds"], rel_tol=1e-12),
                f"{label}: checked median disagrees with per-repetition component sums")
    return point


def report_folder(package, robot, run_layout="packaged"):
    require(run_layout in ("packaged", "full-v3"), "Unsupported input layout")
    return package / (robot if run_layout == "packaged" else f"full-{robot}-v3")


def extract_data(package, article, *, run_layout="packaged"):
    result = {"schema_version": 3, "protocol_version": 2, "execution_revision": "parallel-host-validation-v1",
              "article": article, "task": "two_cube_pick_place_into_box",
              "environment_counts": COUNTS, "statistic": "median",
              "range": "observed minimum and maximum of five measured repetitions; not a confidence interval",
              "warmups_excluded": 1, "simulated_seconds_per_environment": 40,
              "control_frames": 2000, "physics_steps": 40000, "observation_fields": 61,
              "physics_timestep_seconds": .001, "substeps_per_control_frame": 20,
              "observation_dtype": "float32", "reports": {}, "points": []}
    source_hashes, packages = set(), []
    for robot in ROBOTS:
        folder = report_folder(package, robot, run_layout)
        path = folder / "results.json.gz"
        if not path.exists():
            path = folder / "results.json"
        report, provenance = read_report(path)
        config = report["configuration"]
        check_publication_designation((report, config), robot)
        require(config.get("task") == "box" if article == 2 else config.get("scope") == "newton-batch",
                f"{robot}: historical stacking is excluded; expected box benchmark configuration")
        require(report.get("finished_utc"), f"{robot}: report is not finished")
        require(config.get("robot") == robot and config.get("worlds") == COUNTS, f"{robot}: wrong robot or incomplete nine-size coverage")
        require(config.get("repeats") == 5 and config.get("warmups") == 1 and config.get("cpu_threads") == 32,
                f"{robot}: expected five repetitions, one warm-up and 32-worker cap")
        if article == 2:
            require(config.get("cpu_baseline") == "pool", f"{robot}: expected CPU process-pool baseline")
        require(len(report.get("cases", [])) == 18, f"{robot}: expected exactly 18 CPU/GPU configurations")
        identities = report["source"]["file_sha256"]
        aggregate = sha256(json.dumps(identities, sort_keys=True).encode())
        require(aggregate == report["source"]["aggregate_sha256"], f"{robot}: inconsistent source identity")
        source_hashes.add(aggregate)
        versions = report["hardware"]["packages"]
        require(all(versions.get(k) == v for k, v in VERSIONS[article].items()), f"{robot}: different pinned physics stack")
        packages.append(versions)
        provenance.update(started_utc=report["started_utc"], finished_utc=report["finished_utc"],
                          cpu_model=report["hardware"]["cpu_model"], gpu=report["cuda"], packages=versions,
                          source_aggregate_sha256=aggregate, measurement_label=report.get("measurement_label"))
        result["reports"][robot] = provenance
        for worlds in COUNTS:
            pair = []
            for series, backend in BACKENDS[article].items():
                cases = [c for c in report["cases"] if c.get("config", {}).get("backend") == backend and c["config"].get("worlds") == worlds]
                require(len(cases) == 1, f"{robot}/{backend}/{worlds}: expected exactly one case")
                pair.append(extract_case(cases[0], robot, series, worlds, article))
                require(cases[0]["config"].get("validation_workers") == config["cpu_threads"],
                        f"{robot}/{backend}/{worlds}: host validation did not receive the shared requested worker budget")
                if pair[-1]["accepted"]:
                    case_sources = cases[0]["workload"].get("source_sha256", {})
                    require(case_sources and all(identities.get(f"part{article}/{name}") == digest
                            for name, digest in case_sources.items()),
                            f"{robot}/{backend}/{worlds}: case source differs from report source")
                if series == "gpu" and pair[-1]["accepted"]:
                    require(cases[0]["config"]["device"] == report["cuda"].get("alias"),
                            f"{robot}/{worlds}: selected CUDA device differs from reported hardware")
            if all(p["accepted"] for p in pair):
                require(pair[0]["comparison_signature"] == pair[1]["comparison_signature"]
                        and pair[0]["acceptance"] == pair[1]["acceptance"], f"{robot}/{worlds}: CPU/GPU task or recording protocols differ")
                require(pair[0]["backend_settings"]["native_requested"] == pair[1]["backend_settings"]["native_requested"],
                        f"{robot}/{worlds}: CPU/GPU native requested options differ")
                require(pair[0]["host_validation"]["policy"] == pair[1]["host_validation"]["policy"]
                        and pair[0]["host_validation"]["actual"] == pair[1]["host_validation"]["actual"],
                        f"{robot}/{worlds}: CPU/GPU host-validation workers or policy differ")
                if article == 2:
                    require(pair[0]["model_sha256"] == pair[1]["model_sha256"], f"{robot}/{worlds}: native/GPU models differ")
            result["points"].extend(pair)
    require(len(source_hashes) == 1 and packages[0] == packages[1], "Robot reports use different source snapshots or package versions")
    hardware = list(result["reports"].values())
    require(hardware[0]["cpu_model"] == hardware[1]["cpu_model"]
            and hardware[0]["gpu"] == hardware[1]["gpu"], "Robot reports use different hardware")
    result["source_aggregate_sha256"] = source_hashes.pop()
    result["recording_scope"] = "61 float32 physical observations per world at 50 Hz: time, actual opening, final-frame jaw clearance, and each cube's eight corners, maximum corner speed, two jaw-contact counts and two solved jaw normal forces. Both devices use this format."
    result["timer_scope"] = {"simulation": "Episode reset, commands, 40000 physics steps, per-frame backend forward and actual geometry/contact recording, including CPU dispatch/completion or CUDA synchronization. Final host collection/download and validation are separate.",
                             "checked": "Per-repetition simulation_seconds + output_transfer_seconds + validation_seconds. Sum components within each repetition, then compute min/median/max. This includes the time to checked host observations, including validation history copy, dispatch and result collection, but excludes model/tape preparation, setup, compilation, warm-up and validator teardown."}
    result["comparison_scope"] = "Within this article's pinned stack. Both articles use the box task and observation format, but their physics stacks differ; no cross-article speedup is measured."
    result["clock_scope"] = "Protocol v2: independently counted 40000 integration steps per world. Every recorded float32 frame timestamp is checked against sequential additions in the actual backend precision (CPU float64, GPU float32), within one observation ULP. Raw clock values are preserved; nominal task event time is used only after raw-clock and integration-count checks pass."
    result["backend_option_scope"] = "Actual native and uploaded GPU options are retained per accepted case. Pinned MuJoCo Warp clamps requested tolerance to at least 1e-6 and represents numerical options in float32. Solver/integrator/cone choices and collision flags must match the native request; no collider flag override is permitted."
    result["host_validation_scope"] = "Execution revision v3 applies the unchanged serial physical/clock/count oracle independently per world. CPU and GPU cases share the requested host-validation worker budget, capped by worlds and CPU affinity. One worker runs in the parent; larger batches use a persistent spawn pool and read-only shared histories. validation_seconds includes the shared-history copy, dispatch, all oracle work and result collection; validator setup/teardown are separately recorded case costs and excluded from both plotted metrics."
    return result


def date_label(data):
    starts = [datetime.fromisoformat(r["started_utc"].replace("Z", "+00:00")) for r in data["reports"].values()]
    ends = [datetime.fromisoformat(r["finished_utc"].replace("Z", "+00:00")) for r in data["reports"].values()]
    first, last = min(starts).date(), max(ends).date()
    return first.strftime("%d %B %Y").lstrip("0") + ("" if first == last else " – " + last.strftime("%d %B %Y").lstrip("0")) + " UTC"


def caption(data, metric):
    exclusions = "; ".join(f"{ROBOTS[p['robot']]} {p['series'].upper()} at {p['worlds']} environments" for p in data["points"] if not p["accepted"]) or "none"
    hardware = next(iter(data["reports"].values()))
    versions = "; ".join(f"{k} {v}" for k, v in VERSIONS[data["article"]].items())
    return (f"Blog {data['article']}: CPU/GPU scaling for both cubes picked, carried and released into the receiving box. "
            "Each environment runs the complete 40-second, 40000-step task. "
            f"Displayed metric: {'simulation time' if metric == 'simulation' else 'time to checked host results'}. "
            f"{data['timer_scope'][metric]} "
            "Each point is the median of five accepted measured repetitions after one accepted warm-up; whiskers show observed minimum and maximum, not confidence intervals. "
            "Both axes use logarithmic scaling; x positions reflect actual environment counts. CPU uses min(32, environments) workers; GPU uses one device. "
            f"{data['recording_scope']} "
            f"{data['clock_scope']} {data['backend_option_scope']} {data['host_validation_scope']} "
            "Every warm-up and repetition passes physical CubeEvidence checks for both cubes in every environment, including loaded bilateral contact, pickup, retained carry, release, whole-corner containment, settling and gripper withdrawal. "
            f"Excluded cases (no accepted time): {exclusions}. Lines stop at excluded counts; no partial-run timing is plotted. "
            f"Measured {date_label(data)} on shared {hardware['cpu_model']} / {hardware['gpu'].get('name', 'recorded GPU')} hardware. "
            f"Physics versions: {versions}. {data['comparison_scope']} Community task measurements.\n")


def render(data, out, metric):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blog-scaling-matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    except ImportError as error:
        raise SystemExit("Use a separate optional matplotlib environment; do not change the physics lock.") from error
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#88919B", "axes.labelcolor": "#26313C", "text.color": "#182634",
        "xtick.color": "#45515C", "ytick.color": "#45515C", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "johnnys-box-scaling-v2",
        "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), sharey=True)
    fig.subplots_adjust(left=.075, right=.975, bottom=.315, top=.755, wspace=.17)
    article = data["article"]
    title = "Box task: CPU / GPU simulation time" if metric == "simulation" else "Box task: time to checked host results"
    stack = "Native MuJoCo and MuJoCo Warp" if article == 2 else "Newton CPU and Newton CUDA"
    fig.text(.075, .96, title, fontsize=19, weight="bold", va="top")
    fig.text(.075, .914, f"Blog {article}  ·  {stack}  ·  40 simulated seconds per environment", fontsize=11.5)
    h = next(iter(data["reports"].values()))
    cpu = h["cpu_model"].replace("AMD Ryzen ", "").replace(" 64-Cores", "")
    gpu = h["gpu"].get("name", "GPU").replace("NVIDIA ", "").replace(" Workstation Edition", "")
    fig.text(.075, .882, f"{date_label(data)}  ·  Shared {cpu} + {gpu}", fontsize=9.3, color="#596573")
    fig.legend(handles=[Line2D([], [], color=COLORS["cpu"], marker="o", linewidth=2, markersize=6, label="CPU · up to 32 workers"),
                        Line2D([], [], color=COLORS["gpu"], marker="s", linestyle="--", linewidth=2, markersize=6, label="GPU · one device")],
               loc="upper left", bbox_to_anchor=(.067, .858), ncol=2, handlelength=2.5, columnspacing=2.4, fontsize=10.5)
    accepted = [p for p in data["points"] if p["accepted"]]
    require(accepted, "No accepted timings to render")
    low = min(p[metric]["min_seconds"] for p in accepted) / 1.4
    high = max(p[metric]["max_seconds"] for p in accepted) * 1.4
    for ax, (robot, name) in zip(axes, ROBOTS.items()):
        ax.set_title(name, loc="left", pad=11)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlim(.75, 2750); ax.set_ylim(low, high)
        ax.xaxis.set_major_locator(FixedLocator(COUNTS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ticks = [base * 10 ** power for power in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1) for base in (1, 2, 5) if low <= base * 10 ** power <= high]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(which="major", axis="y", color="#E1E5E9", linewidth=.8)
        ax.tick_params(axis="x", labelsize=9.5, rotation=40, pad=4); ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel("Environments per batch (log scale)", labelpad=9)
        for series, marker, line in (("cpu", "o", "-"), ("gpu", "s", "--")):
            points = [p for p in data["points"] if p["robot"] == robot and p["series"] == series]
            ax.plot(COUNTS, [p[metric]["median_seconds"] if p["accepted"] else math.nan for p in points], color=COLORS[series], marker=marker,
                    linestyle=line, linewidth=2, markersize=5.5, markeredgecolor="white", markeredgewidth=.7, zorder=4)
            passed = [p for p in points if p["accepted"]]
            if passed:
                ax.errorbar([p["worlds"] for p in passed], [p[metric]["median_seconds"] for p in passed],
                    yerr=[[p[metric]["median_seconds"] - p[metric]["min_seconds"] for p in passed], [p[metric]["max_seconds"] - p[metric]["median_seconds"] for p in passed]],
                    fmt="none", ecolor=COLORS[series], elinewidth=1.2, capsize=3, capthick=1.2, zorder=3)
        excluded = [p for p in data["points"] if p["robot"] == robot and not p["accepted"]]
        note = "\n".join(f"Excluded {series.upper()}: " + ", ".join(str(p["worlds"]) for p in excluded if p["series"] == series)
                         for series in ("cpu", "gpu") if any(p["series"] == series for p in excluded)) or "All 18 configurations accepted"
        ax.text(0, -.245, note, transform=ax.transAxes, va="top", fontsize=9.3, color="#5C4A3D")
    axes[0].set_ylabel("Median batch wall time (seconds, log scale)", labelpad=10)
    fig.text(.075, .14, "Median of 5 accepted repetitions after 1 warm-up; whiskers show min–max (not confidence intervals).", fontsize=9.5)
    timing = ("Timed: reset, commands, 40,000 steps and state/contact recording; download and validation are separate."
              if metric == "simulation" else "Timed: simulation + host collection/download + task validation, summed within each repetition.")
    fig.text(.075, .105, timing, fontsize=9.5)
    fig.text(.075, .07, "Setup and warm-up are separate. Failed cases have no accepted time and appear as gaps.", fontsize=9.5, color="#596573")
    stem = STEMS[metric]
    description = caption(data, metric)
    meta = {"Title": f"Blog {article} two-cube box scaling: {metric}", "Creator": Path(__file__).name}
    fig.savefig(out / f"{stem}.png", dpi=300, metadata={"Title": meta["Title"], "Description": description})
    fig.savefig(out / f"{stem}.pdf", metadata={**meta, "Subject": description, "CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{stem}.svg", metadata={**meta, "Description": description, "Date": None})
    svg = out / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    (out / f"{stem}.caption.txt").write_text(description)
    plt.close(fig)
    data["renderer"] = {"matplotlib": matplotlib.__version__, "png_dpi": 300, "figure_inches": [12.8, 7.2],
                        "x_scale": "log2", "y_scale": "log10", "colors": COLORS, "script_sha256": sha256(Path(__file__).read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--article", type=int, choices=(2, 3))
    parser.add_argument("--run-layout", choices=("packaged", "full-v3"), default="packaged",
                        help="Read robot/ reports or explicit full-robot-v3/ run folders")
    parser.add_argument("--metric", choices=("both", "simulation", "checked"), default="both")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    package = args.package_dir
    article = args.article
    if article is None:
        candidates = [package.resolve()] if package else []
        candidates += list((package or here).resolve().parents)
        for candidate in candidates:
            if candidate.name in ("Article_2", "Article_3", "article2", "article3"):
                article = int(candidate.name[-1]); break
    require(article in (2, 3), "Specify --article 2 or --article 3")
    if package is None:
        package = here.parent if here.name == "figures" else here / f"article{article}"
    data = extract_data(package, article, run_layout=args.run_layout)
    metrics = ["simulation", "checked"] if args.metric == "both" else [args.metric]
    result = {"article": article, "accepted_points": sum(p["accepted"] for p in data["points"]),
              "excluded_points": sum(not p["accepted"] for p in data["points"]), "metrics": metrics}
    if not args.validate_only:
        require(any(p["accepted"] for p in data["points"]), "No accepted timings to render")
        out = args.output_dir or package / "figures"
        out.mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            render(data, out, metric)
        (out / "fig_box_cpu_gpu_scaling.data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        result["output_dir"] = str(out)
    result["validation_only"] = args.validate_only
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
