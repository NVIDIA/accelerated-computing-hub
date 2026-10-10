#!/usr/bin/env python3
"""Run an explicit, immutable benchmark plan independently of SSH."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    plan_path = Path(sys.argv[1]).resolve()
    plan = json.loads(plan_path.read_text())
    folder = plan_path.parent
    with Path(plan.get("lock_file", str(folder / "workstation-benchmark.lock"))).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = folder / (plan_path.stem + ".status.json")
        if status_path.exists():
            raise RuntimeError("This plan already has a status file; preserve it and choose a new plan")
        status = {"status": "running", "pid": os.getpid(), "started_utc": now(),
                  "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "jobs": []}
        save(status_path, status)
        for job in plan["jobs"]:
            output = Path(job["output_dir"])
            if (output / "results.json").exists() or list(output.glob("*.config.json")):
                raise RuntimeError(f"Prior benchmark files exist: {output}")
            if job.get("prepare_output_dir", True):
                output.mkdir(parents=True, exist_ok=True)
                cache = output / ".warp-cache"
                if not cache.exists():
                    cache.symlink_to(job["warp_cache"], target_is_directory=True)
            record = {"id": job["id"], "started_utc": now(), "status": "running",
                      "command": job["command"], "output_dir": str(output)}
            status["jobs"].append(record)
            log_path = folder / (job["id"] + ".launch.log")
            record["launch_log"] = str(log_path)
            environment = os.environ.copy()
            environment.update(job.get("environment", {}))
            with log_path.open("x") as log:
                child = subprocess.Popen(job["command"], cwd=job["cwd"], env=environment,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                record["pid"] = child.pid
                save(status_path, status)
                record["returncode"] = child.wait()
            record["finished_utc"] = now()
            record["status"] = "passed" if record["returncode"] == 0 else "nonzero_exit"
            report_path = output / "results.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                record["report_status"] = report.get("status")
                record["completed_cases"] = len(report.get("cases", []))
                record["source_aggregate_sha256"] = report.get("source", {}).get("aggregate_sha256")
            save(status_path, status)
            if record["returncode"] and plan.get("stop_on_nonzero", True):
                status["status"] = "stopped_after_nonzero_exit"
                break
        else:
            status["status"] = "finished"
        status["finished_utc"] = now()
        save(status_path, status)


if __name__ == "__main__":
    main()
