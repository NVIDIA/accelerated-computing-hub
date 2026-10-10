#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Reproducible task measurements, isolated workers, and conservative comparisons.

This is a community workload benchmark, not an official NVIDIA product result.
The default Newton batch workload picks both cubes and places them inside a
receiving box. Explicit legacy replay/workflow scopes retain the stacking task.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import signal
import statistics
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
PACKAGES = ("mujoco", "mujoco-warp", "warp-lang", "newton", "numpy")
PRESETS = {"developer": [1, 16], "workstation": [1, 16, 32, 64, 128, 256, 512, 1024, 2048]}


def executable_name(command: str) -> str:
    """nvidia-smi may expose arguments; retain only the executable basename."""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return Path(parts[0]).name if parts else "unknown"


def command_output(command: list[str], timeout: int = 10) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run_isolated_worker(command, *, environment, log, timeout):
    """Bound the entire worker tree, including persistent CPU pool children."""
    if os.name != "posix":
        raise RuntimeError("Isolated benchmark worker groups require Linux or macOS")
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        # The case worker handles TERM through Python finally blocks, which
        # close its pool and unlink its shared history before exiting.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Kill any descendant that outlived the direct worker, then reap
            # that worker before the next timed case can start.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def _worker_terminate(signum, frame):
    raise SystemExit(128 + signum)


def load_snapshot() -> dict:
    """Record contention without collecting user names or process command lines."""
    load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    output = command_output([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,"
        "utilization.gpu,temperature.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits",
    ])
    columns = ("index", "name", "driver", "memory_total_mib", "memory_used_mib",
               "utilization_percent", "temperature_c", "power_w", "sm_clock_mhz")
    gpus = [dict(zip(columns, (part.strip() for part in row)))
            for row in csv.reader(output.splitlines())] if output else []
    processes = command_output([
        "nvidia-smi", "--query-compute-apps=process_name,used_gpu_memory", "--format=csv,noheader,nounits",
    ])
    apps = [{"executable": executable_name(row[0]), "gpu_memory_mib": row[1].strip()}
            for row in csv.reader(processes.splitlines()) if len(row) == 2] if processes else []
    return {"load_average_1_5_15_minutes": load, "gpus": gpus, "gpu_compute_processes": apps}


def hardware_metadata() -> dict:
    cpu = platform.processor()
    physical_cores = None
    memory = None
    if sys.platform.startswith("linux"):
        proc = Path("/proc/cpuinfo")
        if proc.exists():
            cpu = next((line.split(":", 1)[1].strip() for line in proc.read_text().splitlines()
                        if line.startswith("model name")), cpu)
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            memory = next((int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                           if line.startswith("MemTotal:")), None)
        topology = command_output(["lscpu", "-p=CORE,SOCKET"])
        if topology:
            physical_cores = len({line for line in topology.splitlines() if not line.startswith("#")})
    elif sys.platform == "darwin":
        cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
        physical = command_output(["sysctl", "-n", "hw.physicalcpu"])
        physical_cores = int(physical) if physical and physical.isdecimal() else None
        ram = command_output(["sysctl", "-n", "hw.memsize"])
        memory = int(ram) if ram and ram.isdecimal() else None
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    limits = {}
    for name in ("cpu.max", "memory.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            limits[name] = path.read_text().strip()
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "os": platform.system(), "os_release": platform.release(), "architecture": platform.machine(),
        "python": platform.python_version(), "cpu_model": cpu, "physical_cpu_cores": physical_cores,
        "logical_cpu_count": os.cpu_count(), "cpu_affinity": affinity, "ram_bytes": memory,
        "cgroup_limits": limits, "packages": versions,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "PXR_WORK_THREAD_LIMIT", "MUJOCO_GL")},
        "load": load_snapshot(),
    }


def source_identity() -> dict:
    """Hash the actual code used, including uncommitted benchmark changes."""
    selected = list(HERE.glob("*.py")) + list((HERE / "solutions").glob("*.py"))
    selected += [HERE.parent / "requirements.lock.txt"]
    hashes = {str(path.relative_to(HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(selected) if path.is_file()}
    commit = command_output(["git", "-C", str(HERE), "rev-parse", "HEAD"])
    status = command_output(["git", "-C", str(HERE), "status", "--porcelain", "--", ".", "../requirements.lock.txt"])
    return {"git_commit": commit, "dirty": bool(status) if status is not None else None, "file_sha256": hashes,
            "aggregate_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def cuda_preflight(device: str, environment: dict) -> dict:
    script = """import json
try:
 import warp as wp
 wp.init()
 d=wp.get_device(DEVICE)
 print(json.dumps({'available':bool(d.is_cuda),'alias':d.alias,'name':d.name}))
except Exception as e:
 print(json.dumps({'available':False,'reason':str(e)}))
""".replace("DEVICE", repr(device))
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                env=environment, timeout=60)
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {"available": False, "reason": str(exc)}


def newton_batch_memory_preflight(args) -> dict | None:
    """Exact history allocation sizes, excluding solver and temporary storage."""
    if args.scope != "newton-batch":
        return None
    rows = []
    for worlds in args.worlds:
        history = worlds * 2000 * 61 * 4
        rows.append({"worlds": worlds, "history_bytes_per_backend": history,
                     "cpu_shared_history_bytes": history, "cpu_host_result_copy_bytes": history,
                     "gpu_device_history_bytes": history, "gpu_host_download_bytes": history,
                     "validation_shared_history_bytes": history if min(worlds,args.cpu_threads)>1 else 0,
                     "within_trajectory_budget": history <= args.max_trajectory_mib * 1024**2})
    return {"scope": "newton-batch", "task": "two_cube_pick_place_into_box",
            "recording": "2000 frames, 61 float32 values per world; live geometry and solved grasp forces",
            "excludes": "Newton/MuJoCo solver allocations, worker processes, temporary buffers and allocator caches",
            "cpu_shared_memory": "POSIX shared memory; available /dev/shm space is checked by the CPU case",
            "trajectory_budget_mib": args.max_trajectory_mib,
            "within_trajectory_budget": all(row["within_trajectory_budget"] for row in rows),
            "cases": rows}


def case_plan(args) -> list[dict]:
    base = {"robot": args.robot, "frames": 600, "substeps": 10, "repeats": args.repeats,
            "warmups": args.warmups, "device": args.device, "validation_workers": args.cpu_threads,
            "max_trajectory_mib": args.max_trajectory_mib}
    cases = []
    if args.scope in ("replay", "both"):
        for worlds in args.worlds:
            for threads in sorted({1, min(args.cpu_threads, worlds)}):
                cases.append({**base, "scope": "replay", "backend": "mujoco", "worlds": worlds,
                              "cpu_threads": threads, "nconmax": args.nconmax, "njmax": args.njmax})
            if not args.cpu_only:
                cases.append({**base, "scope": "replay", "backend": "mjwarp", "worlds": worlds,
                              "cpu_threads": 1, "nconmax": args.nconmax, "njmax": args.njmax})
    if args.scope == "newton-batch":
        base.update(task="box", frames=2000, substeps=20)
        for worlds in args.worlds:
            cases.append({**base, "scope": "newton-batch", "backend": "newton_cpu", "worlds": worlds,
                          "cpu_threads": min(args.cpu_threads, worlds),
                          "nconmax": args.nconmax, "njmax": args.njmax})
            if not args.cpu_only:
                cases.append({**base, "scope": "newton-batch", "backend": "newton_cuda", "worlds": worlds,
                              "cpu_threads": 1, "nconmax": args.nconmax, "njmax": args.njmax})
    if args.scope in ("workflow", "both"):
        backends = ["mujoco", "newton_cpu"] + ([] if args.cpu_only else ["newton_cuda"])
        for backend in backends:
            cases.append({**base, "scope": "workflow", "backend": backend, "worlds": 1,
                          "cpu_threads": 1, "nconmax": None, "njmax": None})
    random.Random(args.order_seed).shuffle(cases)
    for case in cases:
        case["case_id"] = f"{case['scope']}-{case['backend']}-w{case['worlds']}-t{case['cpu_threads']}"
    return cases


def summarize_row(row: dict) -> dict:
    """Never let a fast failed/partial run become a performance result."""
    config = row.get("config", {})
    samples = row.get("samples", [])
    warmups = row.get("warmup_samples", [])
    required_warmups = config.get("warmups", 0)
    worlds = int(config.get("worlds", row.get("worlds", 0)))
    complete = row.get("status") == "passed" and len(samples) == config.get("repeats") and bool(samples)
    complete = complete and len(warmups) == required_warmups and worlds > 0 and all(
        sample.get("passed") is True and sample.get("task_success_count") == worlds
        and sample.get("task_total_count") == worlds
        and isinstance(sample.get("simulation_seconds"), (int, float))
        and math.isfinite(sample["simulation_seconds"]) and sample["simulation_seconds"] > 0
        and all(isinstance(sample.get(key), (int, float))
                and math.isfinite(sample[key]) and sample[key] >= 0
                for key in ("output_transfer_seconds", "validation_seconds"))
        for sample in [*warmups, *samples])
    summary = {"eligible": bool(complete), "repetitions": len(samples)}
    if not complete:
        return summary
    times = [sample["simulation_seconds"] for sample in samples]
    median = statistics.median(times)
    summary.update({"median_seconds": median, "min_seconds": min(times), "max_seconds": max(times),
                    "successful_tasks_per_second": worlds / median,
                    "world_steps_per_second": worlds * config["frames"] * config["substeps"] / median,
                    "aggregate_real_time_factor": worlds * config["frames"] / 50 / median})
    # This includes measured download/acceptance costs, still excluding scene and tape setup.
    completed = [sample["simulation_seconds"] + sample.get("output_transfer_seconds", 0)
                 + sample.get("validation_seconds", 0) for sample in samples]
    if all(math.isfinite(value) and value > 0 for value in completed):
        summary["median_with_transfer_and_validation_seconds"] = statistics.median(completed)
    return summary


def actual_cpu_threads(row: dict) -> int:
    info = row.get("device_info", {})
    return info.get("cpu_workers", info.get("cpu_threads", row["config"]["cpu_threads"]))


def replay_comparisons(rows: list[dict]) -> list[dict]:
    comparisons = []
    for gpu in rows:
        cfg = gpu.get("config", {})
        if cfg.get("scope") != "replay" or cfg.get("backend") != "mjwarp":
            continue
        candidates = [row for row in rows if row.get("config", {}).get("scope") == "replay"
                      and row.get("config", {}).get("backend") == "mujoco"
                      and row["config"].get("worlds") == cfg.get("worlds")
                      and row["config"].get("robot") == cfg.get("robot")
                      and row.get("summary", {}).get("eligible")]
        if not gpu.get("summary", {}).get("eligible") or not candidates:
            continue
        # A timing ratio is only meaningful for identical model/tape signatures.
        signature = gpu.get("workload", {}).get("comparison_signature")
        candidates = [row for row in candidates if signature and
                      row.get("workload", {}).get("comparison_signature") == signature]
        if not candidates:
            continue
        cpu = min(candidates, key=lambda row: row["summary"]["median_seconds"])
        speedup = cpu["summary"]["median_seconds"] / gpu["summary"]["median_seconds"]
        comparisons.append({"worlds": cfg["worlds"], "cpu_case": cpu["case_id"], "gpu_case": gpu["case_id"],
                            "cpu_threads": actual_cpu_threads(cpu), "simulation_speedup": speedup,
                            "cpu_median_seconds": cpu["summary"]["median_seconds"],
                            "gpu_median_seconds": gpu["summary"]["median_seconds"]})
        # Also compare the time until checked host observations are available.
        complete_key = "median_with_transfer_and_validation_seconds"
        completed_cpu = min(candidates, key=lambda row: row["summary"][complete_key])
        comparisons[-1].update(
            checked_results_cpu_case=completed_cpu["case_id"],
            checked_results_cpu_threads=actual_cpu_threads(completed_cpu),
            checked_results_speedup=completed_cpu["summary"][complete_key] / gpu["summary"][complete_key])
    return sorted(comparisons, key=lambda row: row["worlds"])


def newton_workflow_comparisons(rows: list[dict]) -> list[dict]:
    """Compare Newton devices separately, retaining model/configuration mismatches.

    This measures the cost of the same application protocol on each backend,
    not identical compiled physics. Newton's per-device model construction can
    produce different inertia frames and floating-point/default fields. Retain
    both model hashes and never use this ratio as a replay crossover.
    """
    comparisons = []
    fields = ("task", "frames", "physics_steps", "simulated_seconds", "control_hz",
              "physics_dt_seconds", "controller", "control_policy", "rendering",
              "contact_detection", "output_policy", "timed_scope", "observation_policy",
              "asset_ref", "source_sha256", "model_dimensions", "solver_options",
              "payload_observations")
    config_fields = ("robot", "worlds", "cpu_threads", "frames", "substeps", "repeats", "warmups")
    for gpu in rows:
        cfg = gpu.get("config", {})
        if cfg.get("scope") != "workflow" or cfg.get("backend") != "newton_cuda":
            continue
        gpu_stats = summarize_row(gpu)
        if not gpu_stats["eligible"]:
            continue
        for cpu in rows:
            cpu_cfg = cpu.get("config", {})
            if (cpu_cfg.get("scope") != "workflow" or cpu_cfg.get("backend") != "newton_cpu"
                    or cpu_cfg.get("robot") != cfg.get("robot")):
                continue
            cpu_stats = summarize_row(cpu)
            if not cpu_stats["eligible"]:
                continue
            mismatches = []
            for key in config_fields:
                if key not in cfg or key not in cpu_cfg or cfg[key] != cpu_cfg[key]:
                    mismatches.append(f"config.{key}")
            if cfg.get("worlds") != 1 or cpu_cfg.get("worlds") != 1:
                mismatches.append("single_world_required")
            cpu_workload, gpu_workload = cpu.get("workload", {}), gpu.get("workload", {})
            for key in fields:
                if (key not in cpu_workload or key not in gpu_workload
                        or cpu_workload[key] is None or gpu_workload[key] is None
                        or cpu_workload[key] != gpu_workload[key]
                        or (key == "source_sha256" and not cpu_workload[key])):
                    mismatches.append(f"workload.{key}")
            for label, workload in (("cpu", cpu_workload), ("gpu", gpu_workload)):
                if (not workload.get("model_sha256") or not workload.get("model_mjb_sha256")
                        or workload["model_sha256"] != workload["model_mjb_sha256"]):
                    mismatches.append(f"{label}.model_identity_missing_or_inconsistent")
            comparison = {
                "robot": cfg["robot"], "worlds": cfg["worlds"],
                "cpu_case": cpu["case_id"], "gpu_case": gpu["case_id"],
                "scope": "single-world Newton workflow, including online controller and observations",
                "cpu_median_seconds": cpu_stats["median_seconds"],
                "gpu_median_seconds": gpu_stats["median_seconds"],
                "cpu_checked_median_seconds": cpu_stats["median_with_transfer_and_validation_seconds"],
                "gpu_checked_median_seconds": gpu_stats["median_with_transfer_and_validation_seconds"],
                "compatible": not mismatches, "mismatches": mismatches,
                "cpu_model_sha256": cpu_workload.get("model_sha256"),
                "gpu_model_sha256": gpu_workload.get("model_sha256"),
                "compiled_models_identical": bool(cpu_workload.get("model_sha256")) and
                    cpu_workload.get("model_sha256") == gpu_workload.get("model_sha256"),
            }
            if not mismatches:
                comparison["workflow_cost_ratio"] = cpu_stats["median_seconds"] / gpu_stats["median_seconds"]
                comparison["checked_results_cost_ratio"] = (
                    comparison["cpu_checked_median_seconds"] / comparison["gpu_checked_median_seconds"])
            comparisons.append(comparison)
    return comparisons


def newton_batch_comparisons(rows: list[dict]) -> list[dict]:
    """Pair equal counts of completed Newton worlds under a shared protocol.

    CPU executes independent Newton solvers in persistent worker processes;
    CUDA uses a replicated Newton model. Compiled backend models may differ,
    so this reports batch execution cost, not identical compiled physics.
    """
    def box_protocol(row):
        config, workload = row.get("config", {}), row.get("workload", {})
        return (not row.get("diagnostic_only", False) and row.get("publication_eligible", True)
                and not workload.get("diagnostic_only", False) and workload.get("publication_eligible", True)
                and config.get("task") == "box" and config.get("frames") == 2000
                and config.get("substeps") == 20
                and workload.get("task") == "two_cube_pick_place_into_box"
                and workload.get("frames") == 2000 and workload.get("physics_steps") == 40000
                and workload.get("physics_dt_seconds") == .001
                and workload.get("acceptance", {}).get("protocol_version") == 2
                and workload.get("simulated_seconds") == 40.0)

    comparisons = []
    for gpu in rows:
        cfg = gpu.get("config", {})
        if cfg.get("scope") != "newton-batch" or cfg.get("backend") != "newton_cuda" or not box_protocol(gpu):
            continue
        gpu_stats = summarize_row(gpu)
        signature = gpu.get("workload", {}).get("comparison_signature")
        if not gpu_stats["eligible"] or not signature or not gpu.get("workload", {}).get("model_mjb_sha256"):
            continue
        candidates = []
        for cpu in rows:
            cpu_cfg = cpu.get("config", {})
            if cpu_cfg.get("scope") != "newton-batch" or cpu_cfg.get("backend") != "newton_cpu" or not box_protocol(cpu):
                continue
            stats = summarize_row(cpu)
            shared = ("robot", "worlds", "frames", "substeps", "repeats", "warmups")
            if (not stats["eligible"] or any(key not in cfg or key not in cpu_cfg
                    or cfg[key] != cpu_cfg[key] for key in shared)
                    or cpu.get("workload", {}).get("comparison_signature") != signature
                    or not cpu.get("workload", {}).get("model_mjb_sha256")):
                continue
            candidates.append((cpu, stats))
        if not candidates:
            continue
        cpu, cpu_stats = min(candidates, key=lambda pair: pair[1]["median_seconds"])
        checked = "median_with_transfer_and_validation_seconds"
        checked_cpu, checked_stats = min(candidates, key=lambda pair: pair[1][checked])
        comparisons.append({
            "robot": cfg["robot"], "worlds": cfg["worlds"],
            "cpu_case": cpu["case_id"], "gpu_case": gpu["case_id"],
            "cpu_workers": actual_cpu_threads(cpu), "cpu_parallelism": "persistent worker processes",
            "cpu_median_seconds": cpu_stats["median_seconds"], "gpu_median_seconds": gpu_stats["median_seconds"],
            "batch_cost_ratio": cpu_stats["median_seconds"] / gpu_stats["median_seconds"],
            "checked_results_cpu_case": checked_cpu["case_id"],
            "checked_results_cpu_workers": actual_cpu_threads(checked_cpu),
            "cpu_checked_median_seconds": checked_stats[checked],
            "gpu_checked_median_seconds": gpu_stats[checked],
            "checked_results_cost_ratio": checked_stats[checked] / gpu_stats[checked],
            "comparison_signature": signature,
            "cpu_model_sha256": cpu["workload"]["model_mjb_sha256"],
            "gpu_model_sha256": gpu["workload"]["model_mjb_sha256"],
            "compiled_models_identical": cpu["workload"]["model_mjb_sha256"] == gpu["workload"]["model_mjb_sha256"],
        })
    return sorted(comparisons, key=lambda row: (row["robot"], row["worlds"]))


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_outputs(folder: Path, report: dict) -> None:
    report["comparisons"] = replay_comparisons(report["cases"])
    report["newton_workflow_comparisons"] = newton_workflow_comparisons(report["cases"])
    report["newton_batch_comparisons"] = newton_batch_comparisons(report["cases"])
    write_json(folder / "results.json", report)
    fields = ["case_id", "scope", "backend", "worlds", "cpu_threads", "status", "repetitions",
              "median_seconds", "min_seconds", "max_seconds", "successful_tasks_per_second",
              "world_steps_per_second", "aggregate_real_time_factor", "median_with_transfer_and_validation_seconds"]
    with (folder / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["cases"]:
            writer.writerow({**row.get("config", {}), "case_id": row["case_id"],
                             "cpu_threads": actual_cpu_threads(row),
                             "status": row.get("status", "failed"), **row.get("summary", {})})
    lines = ["# Task-specific migration benchmark", "",
             "Community measurements on the recorded hardware; not an official product benchmark.", "",
             f"Run status: **{report['status']}**. Robot: **{report['configuration']['robot']}**.", "",
             "| Scope | Backend | Worlds | CPU workers | Status | Median / min–max (s) | Successful tasks/s |",
             "|---|---|---:|---:|---|---|---:|"]
    for row in report["cases"]:
        cfg, stats = row["config"], row.get("summary", {})
        elapsed = (f"{stats['median_seconds']:.4f} / {stats['min_seconds']:.4f}–{stats['max_seconds']:.4f}"
                   if stats.get("eligible") else "—")
        throughput = f"{stats['successful_tasks_per_second']:.3f}" if stats.get("eligible") else "—"
        lines.append(f"| {cfg['scope']} | {cfg['backend']} | {cfg['worlds']} | {actual_cpu_threads(row)} | "
                     f"{row.get('status')} | {elapsed} | {throughput} |")
    lines += ["", "Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. "
              "Model setup, control generation, warm-up, final output transfer and acceptance checks are outside "
              "the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes "
              "float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not "
              "policy inference, learning speed or rendering performance.", "",
              "Workflow: optional single-world online controller, simulation and 50 Hz payload observations. Newton reconstructs "
              "the model and actuation; these rows are separate workflow measurements and do not enter the "
              "identical-model replay speedup calculation.", ""]
    lines += ["## Native MuJoCo CPU vs MuJoCo Warp replay", ""]
    if report["comparisons"]:
        lines += ["| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |",
                  "|---:|---:|---:|---:|"]
        lines += [f"| {row['worlds']} | {row['cpu_threads']} / {row['checked_results_cpu_threads']} | "
                  f"{row['simulation_speedup']:.3f}× | {row['checked_results_speedup']:.3f}× |"
                  for row in report["comparisons"]]
        winning = [row for row in report["comparisons"] if row["simulation_speedup"] > 1]
        lines += ["", (f"First measured batch where GPU simulation was faster: **{winning[0]['worlds']} worlds**. "
                        "This is an observed point in this sweep, not a universal crossover or an official recommendation."
                        if winning else "No GPU simulation advantage was measured at the tested batch sizes.")]
        checked_winning = [row for row in report["comparisons"] if row["checked_results_speedup"] > 1]
        lines += ["", (f"First measured batch where checked host results were available faster on GPU: "
                        f"**{checked_winning[0]['worlds']} worlds** (setup excluded)."
                        if checked_winning else "No GPU advantage including output transfer and validation was measured at the tested batch sizes.")]
    else:
        lines += ["No comparable, fully successful CPU/GPU pair is available; no speedup or crossover is reported."]
    lines += ["", "## Newton CPU vs Newton CUDA", "",
              "Both paths use Newton's SolverMuJoCo and the online task controller. CPU selects native "
              "MuJoCo; CUDA selects MuJoCo Warp. This is a single-world application measurement, including "
              "controller and observation overhead; it does not measure batched Newton or solver coupling.", ""]
    if report["newton_workflow_comparisons"]:
        lines += ["| Robot | Newton CPU (s) | Newton CUDA (s) | CPU / CUDA checked result (s) | Workflow cost ratio (CPU / CUDA) |",
                  "|---|---:|---:|---:|---|"]
        for row in report["newton_workflow_comparisons"]:
            ratio = f"{row['workflow_cost_ratio']:.3f}×" if row["compatible"] else "Withheld: compatibility checks differ"
            lines.append(f"| {row['robot']} | {row['cpu_median_seconds']:.4f} | {row['gpu_median_seconds']:.4f} | "
                         f"{row['cpu_checked_median_seconds']:.4f} / {row['gpu_checked_median_seconds']:.4f} | {ratio} |")
        for row in report["newton_workflow_comparisons"]:
            if not row["compatible"]:
                differences = ", ".join(f"`{field}`" for field in row["mismatches"])
                lines += ["", f"{row['robot']}: paired successful observations only. Differences or missing "
                          f"metadata: {differences}. The report does not infer equivalent physics from task success."]
            elif not row["compiled_models_identical"]:
                lines += ["", f"{row['robot']}: compiled CPU/CUDA model hashes differ and both are retained in "
                          "the report. The ratio compares the same Newton application protocol and settings, "
                          "using each backend's constructed model and state interfaces; it does not "
                          "establish identical compiled physics."]
        lines += ["", "Times are medians of complete episodes after warm-up; setup is excluded. "
                  "A reported ratio is CPU time divided by CUDA time, so less than 1× means CUDA was slower. "
                  "These rows never enter the replay crossover calculation."]
    else:
        lines += ["No fully successful Newton CPU/CUDA pair is available. Run `--scope workflow` or "
                  "`--scope both` on a CUDA machine to measure it; failed or unavailable cases remain unpaired."]
    lines += ["", "## Newton CPU vs Newton CUDA across environment counts", "",
              "CPU runs independent Newton SolverMuJoCo instances in persistent worker processes; CUDA "
              "advances a replicated Newton model. Both pick each cube and release it inside the receiving "
              "box over 40 simulated seconds in every environment. Timed batches include reset, stepping, "
              "forward evaluations and live geometry/solved-contact recording; setup, "
              "warm-up, final transfer and task validation are separately recorded.", ""]
    if report["newton_batch_comparisons"]:
        lines += ["| Environments | CPU processes | Newton CPU (s) | Newton CUDA (s) | CPU/CUDA batch cost ratio | Checked-result ratio |",
                  "|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {row['worlds']} | {row['cpu_workers']} | {row['cpu_median_seconds']:.4f} | "
                  f"{row['gpu_median_seconds']:.4f} | {row['batch_cost_ratio']:.3f}× | "
                  f"{row['checked_results_cost_ratio']:.3f}× |" for row in report["newton_batch_comparisons"]]
        lines += ["", "Ratios divide CPU time by CUDA time: above 1× means CUDA finished the measured batch faster. "
                  "Each pair requires equal environment counts, source/protocol identity and all worlds passing. "
                  "Backend model hashes remain recorded; the ratio does not establish identical compiled physics. "
                  "These rows are separate from the one-world online-controller workflow and native MuJoCo replay."]
    else:
        lines += ["No comparable successful Newton batch pair is available. Select `--scope newton-batch` "
                  "to measure Newton CPU and CUDA at the same environment counts."]
    lines += ["", "Hardware, concurrent load, exact dependency versions, source/model/control hashes, "
              "every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and "
              "the individual case JSON files. Missing or failed cases are never replaced with estimates. "
              "Validate the same workload on a developer-accessible GPU before generalizing workstation results.", ""]
    (folder / "summary.md").write_text("\n".join(lines))


def worker(config_path: Path, output: Path) -> int:
    config = json.loads(config_path.read_text())
    try:
        if config["scope"] == "replay":
            from benchmark_workload import run_case
        elif config["scope"] == "newton-batch":
            from benchmark_newton_batch import run_case
        else:
            from benchmark_workflows import run_case
        row = run_case(config)
    except Exception as exc:
        traceback.print_exc()
        row = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "samples": []}
    row.update(case_id=config["case_id"], config=config)
    row["summary"] = summarize_row(row)
    if row.get("status") == "passed" and not row["summary"]["eligible"]:
        row["status"] = "failed"
        row["error"] = "Incomplete or invalid repetition results; timing cannot be published."
    write_json(output, row)
    return 0 if row["status"] == "passed" else 2


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("so101", "rebot"), default="so101")
    parser.add_argument("--scope", choices=("replay", "workflow", "both", "newton-batch"), default="newton-batch")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="workstation")
    parser.add_argument("--worlds", nargs="+", type=int)
    parser.add_argument("--cpu-threads", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--nconmax", type=int)
    parser.add_argument("--njmax", type=int)
    parser.add_argument("--max-trajectory-mib", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=1800, help="Wall-clock budget per isolated case, in seconds.")
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Record availability/configuration without timing tasks.")
    parser.add_argument("--output-dir", type=Path, default=Path(".generated/migration-benchmark"))
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.worker_config:
        if os.name == "posix":
            signal.signal(signal.SIGTERM, _worker_terminate)
        if args.worker_output is None:
            parser.error("worker output is required")
        return worker(args.worker_config, args.worker_output)
    args.worlds = args.worlds or PRESETS[args.preset]
    positive = [*args.worlds, args.cpu_threads, args.repeats, args.warmups, args.max_trajectory_mib, args.timeout]
    positive += [value for value in (args.nconmax, args.njmax) if value is not None]
    if min(positive) < 1 or len(set(args.worlds)) != len(args.worlds):
        parser.error("worlds must be unique and every count/capacity must be positive")
    if not args.device.startswith("cuda:"):
        parser.error("--device must select a CUDA device; use --cpu-only for CPU measurements")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.scope == "workflow" and (args.nconmax is not None or args.njmax is not None):
        parser.error("capacity overrides apply only to replay; workflow uses the tutorial configuration")
    if not args.preflight and ((folder / "results.json").exists() or list(folder.glob("*.config.json"))):
        parser.error("output directory already contains case files; select a new directory to preserve the prior run")
    environment = os.environ.copy()
    environment["PXR_WORK_THREAD_LIMIT"] = "1"
    environment["WARP_CACHE_PATH"] = str(folder / ".warp-cache")
    hardware = hardware_metadata()
    cuda = cuda_preflight(args.device, environment)
    configuration = {key: value for key, value in vars(args).items()
                     if key not in ("output_dir", "worker_config", "worker_output")}
    report = {"schema_version": SCHEMA_VERSION, "status": "preflight" if args.preflight else "running",
              "measurement_label": "community_task_measurement_not_official_product_benchmark",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "configuration": configuration, "hardware": hardware, "cuda": cuda,
              "worker_environment_overrides": {"PXR_WORK_THREAD_LIMIT": "1", "WARP_CACHE_PATH": "<output-dir>/.warp-cache"},
              "source": source_identity(), "cases": []}
    report["memory_preflight"] = newton_batch_memory_preflight(args)
    write_json(folder / "preflight.json", report)
    if args.preflight:
        print(json.dumps(report, indent=2))
        return 0
    if report["memory_preflight"] and not report["memory_preflight"]["within_trajectory_budget"]:
        parser.error("Newton batch recording exceeds --max-trajectory-mib; inspect preflight.json before timing")
    for config in case_plan(args):
        case_id = config["case_id"]
        config_path, output_path = folder / f"{case_id}.config.json", folder / f"{case_id}.json"
        if output_path.exists():
            parser.error(f"prior case result exists: {output_path.name}; select a fresh output directory")
        write_json(config_path, config)
        print(f"Running {case_id} ({args.repeats} repetitions after {args.warmups} warm-up episodes)", flush=True)
        before = load_snapshot()
        started = time.perf_counter()
        if config["backend"] in ("mjwarp", "newton_cuda") and not cuda.get("available"):
            row = {"case_id": case_id, "config": config, "status": "unavailable", "samples": [],
                   "error": "Requested CUDA device unavailable: " + str(cuda.get("reason", args.device))}
        else:
            with (folder / f"{case_id}.log").open("w") as log:
                try:
                    returncode = run_isolated_worker([sys.executable, str(Path(__file__).resolve()),
                                             "--worker-config", str(config_path), "--worker-output", str(output_path)],
                                            environment=environment, log=log, timeout=args.timeout)
                    row = json.loads(output_path.read_text()) if output_path.exists() else {
                        "case_id": case_id, "config": config, "status": "failed", "samples": [],
                        "error": f"Worker exited {returncode} without a result (see log)."}
                    if returncode != 0 and row.get("status") == "passed":
                        row.update(status="failed", error=f"Worker exited {returncode}.")
                except subprocess.TimeoutExpired:
                    row = {"case_id": case_id, "config": config, "status": "timeout", "samples": [],
                           "error": f"Case exceeded {args.timeout} s; no partial timing is accepted."}
        row.update(process_wall_seconds=time.perf_counter() - started,
                   load_before=before, load_after=load_snapshot())
        row["summary"] = summarize_row(row)
        write_json(output_path, row)
        report["cases"].append(row)
        write_outputs(folder, report)
        print(f"  {row['status']}", flush=True)
    report["status"] = "passed" if all(row["summary"]["eligible"] for row in report["cases"]) else "incomplete_or_failed"
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_outputs(folder, report)
    print(f"{report['status']}: {folder / 'summary.md'}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
