#!/usr/bin/env python3
"""Run serial, bounded CPU/GPU batches with the unchanged frozen-v3 probe."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

COUNTS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
SOURCE_SHA = "3ebe1a83688e0289db5bda1feb61af2c82056861e4f9408b38c36cedd3ebcfb0"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, obj):
    with path.open("x") as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write("\n")


def inventory(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    p.add_argument("--stack", choices=["blog2", "blog3"], required=True)
    p.add_argument("--assets", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", type=Path, default=here / "resource-config.json")
    p.add_argument("--device", help="Explicit CUDA device, e.g. cuda:0")
    p.add_argument("--workers", type=int, help="Requested CPU threads; defaults to at most 32 available CPU cores")
    p.add_argument("--counts", default=",".join(map(str, COUNTS)), help="Subset for diagnostics; default is all nine requested batches")
    p.add_argument("--lock-file", type=Path, help="Shared lock for cooperating benchmark runs")
    p.add_argument("--dry-run", action="store_true", help="Print commands and resource configuration without running or writing")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    workers = args.workers if args.workers is not None else min(config["workers"], affinity)
    device = args.device or config["device"]
    try:
        counts = [int(x) for x in args.counts.split(",")]
    except ValueError:
        p.error("counts must be comma-separated integers")
    if not counts or len(set(counts)) != len(counts) or any(x not in COUNTS for x in counts):
        p.error("counts must be a unique subset of 1,16,32,64,128,256,512,1024,2048")
    if not 1 <= workers <= affinity or not device.startswith("cuda:"):
        p.error("workers must fit current CPU affinity; an explicit cuda device is required")
    if config["warmups"] != 1 or config["repeats"] != 5 or config["steps"] != 0 or config["mode"] != "replay":
        p.error("Publication recipe requires one full warmup and five full replay episodes")
    runner = here / "reference_replay_probe.py"
    if hashlib.sha256(runner.read_bytes()).hexdigest() != SOURCE_SHA:
        p.error("Runner differs from frozen-v3 source")
    root = args.output.expanduser().resolve()
    jobs = []
    for worlds in counts:
        for backend in ("cpu", "gpu"):
            name = f"{args.stack}-{backend}-w{worlds}-aloha-pot"
            command = [sys.executable, "-u", str(runner), "--assets", str(args.assets.expanduser().resolve()),
                       "--stack", args.stack, "--backend", backend, "--worlds", str(worlds), "--workers", str(workers),
                       "--device", device, "--mode", "replay", "--steps", "0", "--warmups", "1", "--repeats", "5",
                       "--graph-steps", str(config["graph_steps"]), "--nconmax", str(config["nconmax"]),
                       "--njmax", str(config["njmax"]), "--output", str(root / name)]
            jobs.append({"name": name, "command": command})
    plan = {"created_utc": stamp(), "stack": args.stack, "counts": counts, "workers": workers, "device": device,
            "cpu_affinity_count": affinity, "source_sha256": SOURCE_SHA, "jobs": jobs,
            "scope": "native MuJoCo versus MuJoCo Warp replay reference; not Newton API or box-task timing"}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    lock_path = args.lock_file or root.parent / "reference-benchmark.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((root / job["name"]).exists() for job in jobs):
            p.error("At least one output exists; preserve it and choose a fresh output root")
        subprocess.run([sys.executable, str(here / "fetch_reference_assets.py"), "--output", str(args.assets.expanduser().resolve()), "--verify-only"], check=True)
        write_new(root / f"{args.stack}-run-plan.json", plan)
        hardware = {"utc": stamp(), "platform": platform.platform(), "machine": platform.machine(),
                    "python": sys.version, "cpu_affinity_count": affinity,
                    "cpu": inventory(["lscpu"]), "memory": inventory(["free", "-b"]),
                    "gpus": inventory(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"])}
        write_new(root / f"{args.stack}-hardware.json", hardware)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", PXR_WORK_THREAD_LIMIT="1")
        with (root / f"{args.stack}-execution.jsonl").open("x") as receipts:
            for job in jobs:
                start = stamp()
                with (root / f"{job['name']}.log").open("x") as log:
                    result = subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
                report = root / job["name"] / "results.json"
                status = json.loads(report.read_text()).get("status") if report.exists() else "no_report"
                receipt = {"name": job["name"], "started_utc": start, "finished_utc": stamp(),
                           "exit_code": result.returncode, "report_status": status}
                receipts.write(json.dumps(receipt) + "\n")
                receipts.flush()
                print(json.dumps(receipt), flush=True)
    print(f"Collect with --device {device} --workers {workers}; preserve every failed report and log.")


if __name__ == "__main__":
    main()
