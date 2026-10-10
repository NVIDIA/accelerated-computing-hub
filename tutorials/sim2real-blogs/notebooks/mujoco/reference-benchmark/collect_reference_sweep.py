#!/usr/bin/env python3
"""Read-only acceptance/aggregation of the exact frozen-v3 36-report sweep.

Writes JSON to stdout. Reports and endpoint artifacts are never modified.
Checks retained evidence, not a rerun of the full-history physical validator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

COUNTS = (1, 16, 32, 64, 128, 256, 512, 1024, 2048)
PINNED = {
    "blog2": {"mujoco": "3.8.0", "mujoco-warp": "3.8.0.3", "warp-lang": "1.15.0"},
    "blog3": {"mujoco": "3.12.0", "mujoco-warp": "3.12.0", "warp-lang": "1.17.0", "newton": "1.6.0"},
}
HASHES = {
    "source_sha256": "3ebe1a83688e0289db5bda1feb61af2c82056861e4f9408b38c36cedd3ebcfb0",
    "scene_sha256": "410e6489f6b0f6744caf1f07ff36bfafa42e8c3fb17fefc1737750547f44078f",
    "replay_sha256": "0a0a082bb1a9d5c35a4a045518c72035869fb4dd044685fc734f125605923d28",
    "control_tape_sha256": "2ea25430581d5df71b8c91b41fa4552b96f2e12a04fba39150dc6cc61f68b20f",
}
OPTIONS = {"timestep": .002, "tolerance": 1e-6, "ls_tolerance": 1e-6,
           "iterations": 100., "ls_iterations": 50., "impratio": 10.,
           "solver": 2., "integrator": 0., "cone": 1.}
HEALTH = {f"joint_{x}_coarse_position_bound" for x in (*range(16), 17)} | {
    "unit_free_ball_quaternions", "free_translation_within_scene_extent", "coarse_velocity_bound",
    "upstream_intended_pot_height", "upstream_intended_lid_height"}
MODEL_FIELDS = set("qpos0 body_mass body_inertia body_pos body_quat jnt_type jnt_bodyid jnt_pos jnt_axis jnt_range geom_type geom_bodyid geom_pos geom_quat geom_size geom_friction geom_solref geom_solimp dof_damping dof_armature actuator_trnid actuator_ctrlrange actuator_gainprm actuator_biasprm".split())


def positive(value, zero=False):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and (value >= 0 if zero else value > 0)


def audit(path, stack, backend, worlds, expected_repeats=5, expected_device="cuda:1", expected_workers=32):
    row = {"stack": stack, "backend": backend, "worlds": worlds, "path": str(path),
           "accepted": False, "issues": [], "metrics": None, "endpoint_artifacts": []}
    if not path.exists():
        row["status"] = "missing"
        return row, None
    raw = path.read_bytes()
    row["report_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        report = json.loads(raw)
    except (ValueError, TypeError) as exc:
        row.update(status="unreadable", issues=[str(exc)])
        return row, None
    row["status"] = report.get("status")
    issues = row["issues"]
    def require(ok, label):
        if not ok:
            issues.append(label)
    try:
        c = report["config"]
        expected = {"stack": stack, "backend": backend, "worlds": worlds, "workers": expected_workers,
                    "device": expected_device, "mode": "replay", "steps": 0, "warmups": 1,
                    "repeats": expected_repeats, "graph_steps": 10, "nconmax": 256, "njmax": 1024}
        for key, value in expected.items():
            require(c.get(key) == value, f"config.{key}")
        for key, value in HASHES.items():
            require(report.get(key) == value, key)
        require(report["versions"] == PINNED[stack], "versions")
        require(report["steps"] == 1001 and report["dt"] == .002, "full replay duration")
        require(report["model_dimensions"] == {"nq": 24, "nv": 23, "nu": 14, "na": 0, "nbody": 26, "ngeom": 204}, "model dimensions")
        require(report["solver_options"] == OPTIONS, "solver options")
        metadata = report.get("backend_metadata", {})
        effective = metadata.get("effective_model", {})
        require(set(effective.get("fields", {})) == MODEL_FIELDS, "effective model fields")
        require(effective.get("options") == dict(OPTIONS, disableflags=0., enableflags=0.), "effective model options")
        if backend == "cpu":
            require(metadata.get("cpu_threads") == min(worlds, expected_workers), "expected persistent CPU thread count")
            require(metadata.get("dtype") == "float64", "native dtype")
        else:
            require(metadata.get("device") == expected_device and metadata.get("dtype") == "float32", "GPU device/dtype")
            require(metadata.get("nconmax") == c["nconmax"] and metadata.get("njmax") == c["njmax"] and
                    metadata.get("naconmax") == worlds * c["nconmax"] and
                    metadata.get("ncollision_capacity") == worlds * c["nconmax"], "GPU allocations match configuration")
            upload = metadata.get("upload_comparison", {})
            require(upload.get("passed") is True, "model upload audit")
            require(set(upload.get("fields", {})) == MODEL_FIELDS, "upload field coverage")
            for key, value in upload.get("fields", {}).items():
                require(value.get("passed") is True or (key == "qpos0" and value.get("status") == "not_exposed_by_backend"), f"upload field {key}")
                if value.get("passed") is True and key in effective.get("fields", {}):
                    native = effective["fields"][key]
                    integer = native["dtype"].startswith("int")
                    require(value.get("actual_shape") == native["shape"] and
                            value.get("actual_dtype") == (native["dtype"] if integer else "float32") and
                            value.get("actual_sha256") == native["native_sha256" if integer else "float32_sha256"], f"upload field {key} quantized identity")
            require(set(upload.get("options", {})) == set(OPTIONS) | {"disableflags", "enableflags"}, "upload option coverage")
            require(all(x.get("passed") is True for x in upload.get("options", {}).values()), "upload options")
            integers = {"iterations", "ls_iterations", "solver", "integrator", "cone", "disableflags", "enableflags"}
            for key, expected in dict(OPTIONS, disableflags=0., enableflags=0.).items():
                entry = upload.get("options", {}).get(key, {})
                actual = entry.get("actual")
                expected_value = int(expected) if key in integers else float(np.float32(1 / math.sqrt(expected) if key == "impratio" else expected))
                wanted = expected_value if key in integers else [expected_value]
                require(actual == wanted and entry.get("native") == expected and
                        entry.get("uploaded_field") == ("impratio_invsqrt" if key == "impratio" else key), f"upload option {key} actual value")
        warmups, samples = report.get("warmups", []), report.get("samples", [])
        require(len(warmups) == 1 and len(samples) == expected_repeats, f"one complete warmup and {expected_repeats} measured episodes")
        row["warmup_count"], row["measured_count"] = len(warmups), len(samples)
        row["failed_checks"] = []
        for index, (kind, sample) in enumerate([("warmup", x) for x in warmups] + [("measured", x) for x in samples]):
            label = f"{kind}-{index:02d}"
            validation = sample.get("validation", {})
            checks = validation.get("checks", {})
            required = HEALTH | {"finite_history", "all_world_step_clocks"}
            if backend == "gpu":
                required |= {"integration_count", "contacts_within_allocation", "constraints_within_allocation", "broadphase_within_allocation"}
                if stack == "blog3":
                    required.add("no_capacity_overflow_flags")
            require(required <= set(checks), f"{label} validation coverage")
            require(validation.get("passed") is True and all(x is True for x in checks.values()), f"{label} verdict")
            row["failed_checks"].extend(f"{label}:{k}" for k, v in checks.items() if v is not True)
            health = validation.get("physical_health") or {}
            require(set(health.get("checks", {})) == HEALTH, f"{label} health coverage")
            require(all(v is True and checks.get(k) is True for k, v in health.get("checks", {}).items()), f"{label} health/check agreement")
            require(health.get("policy", {}).get("upstream_coarse_lift", {}).get("applied") is True, f"{label} full lift check")
            for key in ("simulation_seconds", "output_collection_seconds", "history_transfer_seconds", "validation_seconds"):
                require(positive(sample.get(key), zero=key != "simulation_seconds"), f"{label} {key}")
            if backend == "gpu":
                diag = sample.get("diagnostics", {})
                require(diag.get("integrations") == 1001, f"{label} integration count")
                for metric, capacity in (("peak_total_contacts", "naconmax"), ("peak_constraints_per_world", "njmax"), ("peak_broadphase_pairs", "ncollision_capacity")):
                    value, bound = diag.get(metric), metadata.get(capacity)
                    require(positive(value, zero=True) and positive(bound) and value <= bound, f"{label} {metric} allocation")
                if stack == "blog3":
                    overflow = diag.get("overflow", [])
                    require(len(overflow) == worlds and all(isinstance(x, int) and not x & ~1536 for x in overflow), f"{label} overflow bits")
                collection = diag.get("collection_timing", {})
                history, other = collection.get("history_transfer_seconds"), collection.get("diagnostic_collection_seconds")
                require(positive(history, zero=True) and positive(other, zero=True) and
                        math.isclose(history + other, sample["output_collection_seconds"], rel_tol=1e-12, abs_tol=1e-12), f"{label} all-gather collection accounting")
                require(history == sample["history_transfer_seconds"], f"{label} history transfer accounting")
            endpoint = path.parent / f"{label}-endpoints.npz"
            require(endpoint.is_file(), f"{label} endpoint evidence")
            if endpoint.is_file():
                eraw = endpoint.read_bytes()
                row["endpoint_artifacts"].append({"path": str(endpoint), "bytes": len(eraw), "sha256": hashlib.sha256(eraw).hexdigest()})
                with np.load(endpoint, allow_pickle=False) as z:
                    require(set(z.files) == {"first_state", "final_state"}, f"{label} endpoint keys")
                    dtype = np.dtype("float64" if backend == "cpu" else "float32")
                    for key, expected_time in (("first_state", .002), ("final_state", 2.002)):
                        a = z[key]
                        require(a.shape == (worlds, 48) and a.dtype == dtype and np.isfinite(a).all(), f"{label} {key} schema/finite")
                        if a.shape == (worlds, 48):
                            target = np.cumsum(np.full(1001, .002, dtype=dtype))[0 if key == "first_state" else -1]
                            require(np.allclose(a[:, 0], target, rtol=1e-6, atol=1e-6), f"{label} {key} clock")
        require(report.get("status") == "complete", "complete report status")
        if report.get("status") != "complete":
            require(all(report.get(k) is None for k in ("median_simulation_seconds", "median_checked_history_seconds", "world_steps_per_second")), "failed timings must be excluded")
        if not issues:
            sim = statistics.median(x["simulation_seconds"] for x in samples)
            checked = statistics.median(x["simulation_seconds"] + x["output_collection_seconds"] + x["validation_seconds"] for x in samples)
            metrics = {"median_simulation_seconds": sim, "median_checked_history_seconds": checked,
                       "world_steps_per_second": worlds * 1001 / sim}
            for key, value in metrics.items():
                require(positive(report.get(key)) and math.isclose(report[key], value, rel_tol=1e-12), f"recomputed {key}")
            if not issues:
                row.update(accepted=True, metrics=metrics)
    except (KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        issues.append(f"schema/artifact error: {type(exc).__name__}: {exc}")
    return row, report


def collect(root, expected_device="cuda:1", expected_workers=32):
    rows, raw = {}, {}
    for stack in PINNED:
        for worlds in COUNTS:
            for backend in ("cpu", "gpu"):
                name = f"{stack}-{backend}-w{worlds}-aloha-pot"
                rows[name], raw[name] = audit(root / name / "results.json", stack, backend, worlds,
                                            expected_device=expected_device, expected_workers=expected_workers)
    pairs = []
    for stack in PINNED:
        for worlds in COUNTS:
            names = [f"{stack}-{b}-w{worlds}-aloha-pot" for b in ("cpu", "gpu")]
            a, b = [rows[n] for n in names]
            pair = {"stack": stack, "worlds": worlds, "accepted": False,
                    "simulation_speedup_cpu_over_gpu": None, "checked_speedup_cpu_over_gpu": None, "issues": []}
            if a["accepted"] and b["accepted"]:
                ca, cb = [raw[n] for n in names]
                for key in ("platform", "machine", "versions", "solver_options", "model_dimensions", *HASHES):
                    if ca[key] != cb[key]:
                        pair["issues"].append(f"pair mismatch: {key}")
                ea, eb = [r["backend_metadata"]["effective_model"] for r in (ca, cb)]
                for key in ("fields", "options", "compiled_model_sha256"):
                    if ea[key] != eb[key]:
                        pair["issues"].append(f"pair effective model mismatch: {key}")
                if not pair["issues"]:
                    pair.update(accepted=True,
                        simulation_speedup_cpu_over_gpu=a["metrics"]["median_simulation_seconds"] / b["metrics"]["median_simulation_seconds"],
                        checked_speedup_cpu_over_gpu=a["metrics"]["median_checked_history_seconds"] / b["metrics"]["median_checked_history_seconds"])
            else:
                pair["issues"].append("both matching reports must pass before comparing timings")
            pairs.append(pair)
    return {"collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "expected_resources": {"device": expected_device, "workers": expected_workers},
            "scope": "adapted ALOHA replay pipeline; native MuJoCo versus MuJoCo Warp; not Newton API or two-cube task validation",
            "limitations": ["Endpoint artifacts and recorded all-history checks audited; intermediate trajectories were not persisted.",
                            "Float64 CPU and float32 GPU retain full state history with different bandwidth cost.",
                            "Coarse physical health and final-height checks do not prove grasp/contact/hold success.",
                            "Hardware inventory/utilization and supervisory run receipts require separate review."],
            "expected_reports": 36, "accepted_reports": sum(x["accepted"] for x in rows.values()),
            "expected_pairs": 18, "accepted_pairs": sum(x["accepted"] for x in pairs),
            "reports": list(rows.values()), "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1", help="Expected selected GPU, matching the run configuration")
    parser.add_argument("--workers", default=32, type=int, help="Expected requested CPU worker count")
    args = parser.parse_args()
    if args.workers < 1 or not args.device.startswith("cuda:"):
        parser.error("Positive workers and an explicit cuda device required")
    result = collect(args.results_root, args.device, args.workers)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
