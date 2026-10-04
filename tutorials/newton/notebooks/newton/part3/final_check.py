#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Final checks for Article 2 and Newton: task outcomes, not just exit codes.

Use from 04_Notebook_Final_Check.ipynb, or from a terminal in part3::

    python final_check.py --skip-gpu --robot both
    python final_check.py --solutions --skip-gpu --robot both --include-clean-table
    python final_check.py --solutions --robot both --include-newton-cuda --device cuda:1

Student entry points are the default. --solutions runs the references directly;
neither mode copies over exercises. Skipped GPU checks are never passes. A True
return means all *executed* checks passed (and at least one executed), not that a
skipped backend or a many-world benchmark was verified. Every subprocess uses
this interpreter; Article 2 checks here are reruns in the Article 3 environment,
not a reproduction of Article 2's separately pinned historical environment.
Clean-table checks also save matched JSON/NPZ pairs under .generated/final_check,
with distinct reference/student and robot names for offline export. Explicit
CUDA selection adds a _cudaN suffix to preserve each GPU's measurements.
--include-newton-cuda additionally runs both Newton CUDA contact paths; --device
selects one CUDA device for all GPU checks. These remain single-world task gates.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PART3_DIR = Path(__file__).resolve().parent
CODE_ROOT = PART3_DIR.parent.parent

# Article 2's physical task gate: alignment within 15 mm, one 44 mm cube high.
MAX_XY_ERROR_M = 0.015
MIN_DZ_M = 0.035
MAX_DZ_M = 0.055
STACK_RE = re.compile(r"\bxy_err=(\S+)\s+(?:m\s+)?dz=(\S+)")
CLEAN_TABLE_PREFIX = "CLEAN_TABLE_RESULT "
CLEAN_TABLE_OBJECTS = frozenset(("red_cube", "blue_cube", "shirt", "cable"))
CLEAN_TABLE_TASK = "gripper_pick_place_into_bin"
CLEAN_TABLE_SCHEMA_VERSION = 2
FRAME_DT = 0.02
MIN_SETTLED_FRAMES = 50  # one simulated second at the task's 50 Hz controller
MIN_TOOL_CLEARANCE_M = 0.02  # lowest measured jaw extent above rim/working plane
MIN_GRASP_FRAMES = 10
MIN_LIFT_FRAMES = 25
MIN_CARRY_FRAMES = 10
MIN_LIFT_CLEARANCE_M = 0.01
MIN_CARRY_DISTANCE_M = 0.05
CONTAINMENT_TOLERANCE_M = 0.003


@dataclass
class Check:
    name: str
    script: Path
    args: list[str]
    requires_cuda: bool = False
    metric: str = "stack"
    robot: str | None = None
    timeout: float = 900
    device: str | None = None
    warp_device: str | None = None
    physics: str | None = None


def checks_for(
    robot: str, *, solutions: bool = False, include_clean_table: bool = False,
    include_newton_cuda: bool = False, device: str | None = None,
) -> list[Check]:
    """Select one robot's student or reference entry points without modifying them."""
    if robot not in ("so101", "rebot"):
        raise ValueError(f"Unknown robot: {robot!r}; choose so101 or rebot")
    if device is not None and re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError(f"Invalid CUDA device: {device!r}; use cuda:N")
    robot_args = ["--robot", robot]
    checks = [
        Check(
            name=f"Article 2 / Part 1 — MuJoCo CPU ({robot})",
            script=CODE_ROOT / "mujoco" / "part1" / "so101_pick_place.py",
            args=["--headless-steps", "600", *robot_args],
        ),
        Check(
            name=f"Article 2 / Part 2 — MJWarp, one CUDA world ({robot})" + (f" [{device}]" if device else ""),
            script=CODE_ROOT / "mujoco" / "part2" / "so101_mjwarp.py",
            args=["--headless-steps", "600", *robot_args],
            requires_cuda=True, warp_device=device,
        ),
        Check(
            name=f"Article 3 / Part 3 — Newton SolverMuJoCo, native CPU ({robot})",
            script=PART3_DIR / "so101_newton.py",
            args=["--viewer", "null", "--device", "cpu", "--num-frames", "600", "--test", *robot_args],
        ),
    ]
    if include_newton_cuda:
        for contacts in ("Newton", "MuJoCo"):
            cuda_device = device or "cuda:0"
            checks.append(Check(
                name=f"Article 3 / Part 3 — Newton SolverMuJoCo, {cuda_device} / {contacts} contacts ({robot})",
                script=PART3_DIR / "so101_newton.py",
                args=["--viewer", "null", "--device", cuda_device, "--num-frames", "600", "--test",
                      *robot_args, *(["--use-mujoco-contacts"] if contacts == "MuJoCo" else [])],
                requires_cuda=True, device=cuda_device,
                physics=f"Physics: MuJoCo Warp (CUDA); contacts: {contacts};",
            ))
    if include_clean_table:
        mode = "reference" if solutions else "student"
        device_suffix = f"_{device.replace(':', '')}" if device is not None else ""
        artifact = PART3_DIR / ".generated" / "final_check" / f"{mode}_clean_{robot}{device_suffix}"
        checks.append(Check(
            name=f"Article 3 — Newton clean-the-table ({robot})",
            script=PART3_DIR / "clean_the_table.py",
            args=["--robot", robot, "--viewer", "null", "--test",
                  "--report", str(artifact.with_suffix(".json")), "--record", str(artifact.with_suffix(".npz"))],
            metric="clean_table", robot=robot, timeout=1800, device=device,
        ))
        if device is not None:
            checks[-1].args.extend(["--device", device])
    if solutions:
        for check in checks:
            check.script = check.script.parent / "solutions" / f"{check.script.stem}_solution.py"
    return checks


def cuda_available() -> bool:
    """Probe actual CUDA availability, not Warp's current default device.

    Import/initialization errors propagate: a broken probe is not proof that a
    machine lacks CUDA. --skip-gpu avoids the probe entirely.
    """
    import warp as wp

    wp.init()
    return wp.is_cuda_available()


def parse_stack(output: str) -> tuple[float, float] | None:
    """Read the final stack report, supporting units and scientific notation."""
    lines = [line for line in output.splitlines() if "xy_err=" in line]
    if not lines:
        return None
    match = STACK_RE.search(lines[-1])
    if match is None:
        return None
    try:
        xy_err, dz = map(float, match.groups())
    except ValueError:
        return None
    if not math.isfinite(xy_err) or not math.isfinite(dz):
        return None
    return xy_err, dz


def validate_clean_table(
    output: str, *, robot: str | None, device: str | None = None,
) -> tuple[bool, str]:
    """Validate the final explicit task report; process success alone proves nothing."""
    reports = [line for line in output.splitlines()
               if line.startswith(CLEAN_TABLE_PREFIX.rstrip())]
    if not reports:
        return False, "no CLEAN_TABLE_RESULT found (run with --test)"
    if not reports[-1].startswith(CLEAN_TABLE_PREFIX):
        return False, "malformed final CLEAN_TABLE_RESULT line"

    def reject_nonfinite(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        report = json.loads(reports[-1][len(CLEAN_TABLE_PREFIX):], parse_constant=reject_nonfinite)
    except (ValueError, TypeError) as exc:
        return False, f"invalid CLEAN_TABLE_RESULT JSON: {exc}"
    return validate_clean_table_report(report, robot=robot, device=device)


def _finite_number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _vector3(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(_finite_number(x) for x in value)


def _bounds(lower, upper) -> bool:
    return _vector3(lower) and _vector3(upper) and all(lo <= hi for lo, hi in zip(lower, upper))


def _contained(lower, upper, bin_lower, bin_upper, axes=3) -> bool:
    return all(bin_lower[i] - CONTAINMENT_TOLERANCE_M <= lower[i]
               and upper[i] <= bin_upper[i] + CONTAINMENT_TOLERANCE_M for i in range(axes))


def validate_clean_table_report(
    report: dict, *, robot: str | None, device: str | None = None,
) -> tuple[bool, str]:
    """Validate measured grasp history and terminal geometry, without physics imports.

    This pure gate is shared by the subprocess runner and offline renderer.
    Historical sweep reports deliberately fail: containment and aggregate
    coupling feedback do not establish a grasp, loaded carry, or release.
    """
    if not isinstance(report, dict) or report.get("success") is not True:
        return False, "clean-table report must contain success=true"
    if "failure" not in report or report["failure"] is not None or report.get("finite") is not True:
        return False, "clean-table needs finite=true and no recorded task failure"
    try:
        json.dumps(report, allow_nan=False)
    except (ValueError, TypeError):
        return False, "clean-table report must contain finite JSON values"
    if (report.get("task") != CLEAN_TABLE_TASK
            or type(report.get("schema_version")) is not int
            or report["schema_version"] != CLEAN_TABLE_SCHEMA_VERSION):
        return False, "clean-table needs gripper_pick_place_into_bin schema_version=2; sweeping is not a grasp task"
    reported_device = report.get("device")
    if (report.get("robot") not in ("so101", "rebot") or report.get("robot") != robot
            or not isinstance(reported_device, str)
            or re.fullmatch(r"cpu|cuda:[0-9]+", reported_device) is None):
        return False, "clean-table robot/device missing, invalid, or wrong robot"
    if device is not None and reported_device != device:
        return False, f"clean-table ran on {reported_device}; expected {device}"
    frames = report.get("frames")
    if type(frames) is not int or not _finite_number(frames) or frames <= 0 or report.get("phase") != "done":
        return False, "clean-table needs positive frames and phase=done"
    seconds = report.get("simulation_seconds")
    if not _finite_number(seconds) or not math.isclose(seconds, frames * FRAME_DT, abs_tol=1e-6):
        return False, "clean-table simulation_seconds must match its 50 Hz frame count"
    bin_lower, bin_upper = report.get("bin_lower"), report.get("bin_upper")
    if not _bounds(bin_lower, bin_upper) or any(lo >= hi for lo, hi in zip(bin_lower, bin_upper)):
        return False, "clean-table needs finite, nonempty bin bounds"
    table_z = report.get("table_z")
    if not _finite_number(table_z):
        return False, "clean-table needs the measured table height"
    clearance = report.get("tool_clearance_m")
    if (report.get("tool_withdrawn") is not True or isinstance(clearance, bool)
            or not isinstance(clearance, (int, float)) or not math.isfinite(clearance)
            or clearance < MIN_TOOL_CLEARANCE_M):
        return False, "clean-table needs measured tool withdrawal with finite clearance >=0.02 m"
    force = report.get("max_coupling_input_force_norm")
    if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) or force <= 1e-5:
        return False, "clean-table needs finite nonzero coupling feedback"
    soft_contacts = report.get("max_soft_contacts")
    if type(soft_contacts) is not int or soft_contacts <= 0:
        return False, "clean-table needs observed soft contacts"
    overflow = report.get("mujoco_warp_overflow_flags")
    # MuJoCo Warp 3.12: ITERATIONS=512, LS_ITERATIONS=1024 are convergence
    # diagnostics. Every other bit must fail, including unknown future flags.
    if type(overflow) is not int or overflow < 0 or overflow & ~(512 | 1024):
        return False, "clean-table report contains missing/invalid MuJoCo Warp capacity flags"
    objects = report.get("objects")
    if not isinstance(objects, dict) or set(objects) != CLEAN_TABLE_OBJECTS:
        return False, "clean-table report must include exactly red_cube, blue_cube, shirt and cable"
    intervals = []
    for name, obj in objects.items():
        if not isinstance(obj, dict) or obj.get("inside") is not True:
            return False, f"clean-table {name}: not inside the bin"
        lower, upper = obj.get("bounds_min"), obj.get("bounds_max")
        if not _bounds(lower, upper) or not _contained(lower, upper, bin_lower, bin_upper):
            return False, f"clean-table {name}: complete final geometry must be inside the bin"
        settled = obj.get("settled_frames")
        if type(settled) is not int or not MIN_SETTLED_FRAMES <= settled <= frames:
            return False, f"clean-table {name}: need at least 50 settled frames, not more than total frames"
        speed = obj.get("max_point_speed")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0.0 <= speed < 0.04:
            return False, f"clean-table {name}: final point speed must be finite and below 0.04 m/s"
        for field in ("initially_outside_bin", "grasped", "lifted", "carried", "over_bin_before_release",
                      "release_commanded", "released"):
            if obj.get(field) is not True:
                return False, f"clean-table {name}: missing measured {field}"
        if obj.get("dropped_before_release") is not False:
            return False, f"clean-table {name}: accepted grasp was lost before intentional release"
        counts = {
            "max_loaded_bilateral_frames": MIN_GRASP_FRAMES,
            "full_lift_frames": MIN_LIFT_FRAMES,
            "carry_frames": MIN_CARRY_FRAMES,
            "detached_settled_frames": MIN_SETTLED_FRAMES,
        }
        for field, minimum in counts.items():
            value = obj.get(field)
            if type(value) is not int or not minimum <= value <= frames:
                return False, f"clean-table {name}: need {minimum} measured {field}, not more than total frames"
        if (obj["full_lift_frames"] > obj["max_loaded_bilateral_frames"]
                or obj["carry_frames"] > obj["max_loaded_bilateral_frames"]
                or obj["detached_settled_frames"] > settled):
            return False, f"clean-table {name}: inconsistent contact/lift/carry/settling windows"
        event_fields = ("grasp_time", "lift_time", "carry_time", "release_command_time", "release_time")
        times = [obj.get(field) for field in event_fields]
        if not all(_finite_number(t) and 0 <= t <= seconds for t in times):
            return False, f"clean-table {name}: grasp/lift/carry/release times must be finite and within the run"
        grasp, lift, carry, command, release = times
        if not grasp < lift <= carry < command <= release:
            return False, f"clean-table {name}: grasp/lift/carry/open/release events are out of order"
        if ((MIN_LIFT_FRAMES - 1) * FRAME_DT > carry - grasp + 1e-6
                or (settled - 1) * FRAME_DT > seconds - release + 1e-6):
            return False, f"clean-table {name}: measured windows do not fit the event times"
        intervals.append((grasp, release, name))
        lift_clearance = obj.get("min_lift_clearance_m")
        if not _finite_number(lift_clearance) or lift_clearance < MIN_LIFT_CLEARANCE_M:
            return False, f"clean-table {name}: sustained whole-object lift must clear the table by 0.01 m"
        force = obj.get("min_jaw_normal_force_N")
        if not _finite_number(force) or force <= 1e-5:
            return False, f"clean-table {name}: loaded carry needs measured force from both jaws"
        carry_lower, carry_upper = obj.get("carry_bounds_min"), obj.get("carry_bounds_max")
        if (not _bounds(carry_lower, carry_upper)
                or not _contained(carry_lower, carry_upper, bin_lower, bin_upper, axes=2)
                or carry_lower[2] < table_z + MIN_LIFT_CLEARANCE_M):
            return False, f"clean-table {name}: whole payload must reach above the bin while lifted"
        grasp_center, carry_center = obj.get("grasp_center"), obj.get("carry_center")
        carry_start = obj.get("carry_start_center")
        if not all(_vector3(value) for value in (grasp_center, carry_start, carry_center)):
            return False, f"clean-table {name}: loaded carry needs measured grasp/start/arrival centers"
        if not all(carry_lower[i] <= carry_center[i] <= carry_upper[i] for i in range(3)):
            return False, f"clean-table {name}: carry center is outside its measured geometry"
        if all(bin_lower[i] <= grasp_center[i] <= bin_upper[i] for i in range(2)):
            return False, f"clean-table {name}: grasp must start outside the receiving bin"
        distance = obj.get("carry_distance_m")
        measured_distance = math.hypot(*(carry_center[i] - carry_start[i] for i in range(2)))
        if (not _finite_number(distance) or distance < MIN_CARRY_DISTANCE_M
                or not math.isclose(distance, measured_distance, rel_tol=1e-5, abs_tol=1e-6)):
            return False, f"clean-table {name}: loaded payload must travel at least 0.05 m in XY"
        opening = obj.get("release_open_fraction")
        if not _finite_number(opening) or not 0.8 <= opening <= 1.0:
            return False, f"clean-table {name}: intentional release needs measured jaw opening >=0.8"
        jaw_contacts = obj.get("jaw_contacts")
        if (not isinstance(jaw_contacts, (list, tuple)) or len(jaw_contacts) != 2
                or any(count is not False and not (type(count) is int and count == 0) for count in jaw_contacts)):
            return False, f"clean-table {name}: settled payload must be detached from both jaws"
    intervals.sort()
    if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
        return False, "clean-table accepted per-object grasp/release intervals overlap"
    return True, (
        f"{len(objects)} objects grasped, lifted, carried, released and settled for >=50 frames; "
        f"tool withdrawn (clearance={clearance:.4f} m); "
        f"phase=done, frames={frames}, device={report['device']}, robot={robot}"
    )


def run_check(check: Check, *, timeout: float | None = None) -> tuple[bool, str]:
    """Run one script with this Python and validate the selected task metric."""
    timeout = check.timeout if timeout is None else timeout
    if not check.script.is_file():
        return False, f"{check.script} not found"
    command = [sys.executable, check.script.name, *check.args]
    if check.warp_device is not None:
        # Article 2 predates a --device option. Select Warp in this child only,
        # retaining physical GPU ordinals and the script's normal argv/cwd.
        command = [sys.executable, "-c", (
            "import runpy, sys; import warp as wp; "
            "wp.init(); wp.set_device(sys.argv[1]); sys.argv = sys.argv[2:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        ), check.warp_device, check.script.name, *check.args]
    try:
        proc = subprocess.run(
            command,
            cwd=check.script.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        tail = output.strip().splitlines()[-8:]
        return False, f"script exited with an error ({exc.returncode}):\n      " + "\n      ".join(tail)
    except OSError as exc:
        return False, f"could not launch script: {exc}"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s; no completed task verified"

    output = proc.stdout + proc.stderr
    if check.metric == "clean_table":
        return validate_clean_table(output, robot=check.robot, device=check.device)
    if check.metric != "stack":
        return False, f"unknown check metric: {check.metric}"
    if check.physics is not None and not any(line.startswith(check.physics) for line in output.splitlines()):
        return False, f"expected physics backend/contact path was not reported: {check.physics}"
    stacked = parse_stack(output)
    if stacked is None:
        return False, "no valid finite final stack check found (did you complete the verification step?)"
    xy_err, dz = stacked
    if not 0.0 <= xy_err <= MAX_XY_ERROR_M:
        return False, f"cubes not aligned: xy_err={xy_err:.6f} m (want 0-{MAX_XY_ERROR_M})"
    if not MIN_DZ_M <= dz <= MAX_DZ_M:
        return False, f"cube not stacked: dz={dz:.6f} m (want {MIN_DZ_M}-{MAX_DZ_M})"
    return True, f"stacked (xy_err={xy_err:.6f} m, dz={dz:.6f} m)"


def run_final_check(
    skip_gpu: bool = False,
    robot: str = "so101",
    *,
    solutions: bool = False,
    include_clean_table: bool = False,
    include_newton_cuda: bool = False,
    device: str | None = None,
) -> bool:
    """Return True only if at least one check ran and every executed check passed.

    GPU skips remain unverified and are counted separately. This gate runs one
    MJWarp world per robot when CUDA is available; optional Newton CUDA checks
    cover both contact paths on the selected device. This is not a scaling study.
    """
    if robot not in ("so101", "rebot", "both"):
        raise ValueError(f"Unknown robot: {robot!r}; choose so101, rebot or both")
    if device is not None and re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError(f"Invalid CUDA device: {device!r}; use cuda:N")
    print(f"Mode: {'reference solutions' if solutions else 'student exercises'}; Python: {sys.executable}")
    if skip_gpu:
        has_cuda = False
        skip_reason = "disabled by request (--skip-gpu); unverified"
    else:
        try:
            has_cuda = cuda_available()
        except Exception as exc:  # a probe error is a failure, never a GPU skip
            print(f"[FAIL] CUDA discovery: {type(exc).__name__}: {exc}")
            print("Fix the environment, or explicitly request --skip-gpu for a CPU-only check.")
            return False
        skip_reason = "no CUDA device detected; unverified"

    robots = ("so101", "rebot") if robot == "both" else (robot,)
    passed = failed = skipped = 0
    for name in robots:
        selection = dict(solutions=solutions, include_clean_table=include_clean_table)
        if include_newton_cuda:
            selection["include_newton_cuda"] = True
        if device is not None and not skip_gpu and has_cuda:
            selection["device"] = device
        for check in checks_for(name, **selection):
            if check.requires_cuda and (skip_gpu or not has_cuda):
                print(f"[SKIP] {check.name}: {skip_reason}")
                skipped += 1
                continue
            # Pin the optional task and validate its reported device. Never let
            # an ambient default turn a requested CPU check into a CUDA run.
            if check.metric == "clean_table":
                selected_device = "cpu" if skip_gpu or not has_cuda else (device or "cuda:0")
                if "--device" in check.args:
                    check.args[check.args.index("--device") + 1] = selected_device
                else:
                    check.args.extend(["--device", selected_device])
                check.device = selected_device
            print(f"[....] {check.name}: running {check.script} ...", flush=True)
            ok, detail = run_check(check)
            passed += int(ok)
            failed += int(not ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {check.name}: {detail}", flush=True)

    print(f"\nSummary: {passed} passed, {failed} failed, {skipped} skipped.")
    if passed and not failed:
        if skipped:
            print("Executed checks passed; skipped checks remain unverified.")
        else:
            print("All selected task checks passed. This does not verify GPU batch scaling or performance.")
        return True
    print("No successful complete executed suite. Read the failures; unfinished student TODOs are not skips.")
    print("Compare with solutions/ or use --solutions explicitly; never overwrite your exercises.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Final task checks: MuJoCo, MJWarp, Newton, optional coupling.")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip CUDA-only checks; force optional Newton demo to CPU.")
    parser.add_argument("--solutions", action="store_true", help="Run references directly instead of student exercises.")
    parser.add_argument("--include-clean-table", action="store_true", help="Also assert the long-running coupled task (1800s timeout per robot).")
    parser.add_argument("--include-newton-cuda", action="store_true", help="Also check Newton CUDA with Newton and MuJoCo contacts.")
    parser.add_argument("--device", help="Select a CUDA device for GPU checks (cuda:N); --skip-gpu forces the optional task to CPU.")
    parser.add_argument("--robot", default="so101", choices=("so101", "rebot", "both"))
    args = parser.parse_args()
    if args.device is not None and re.fullmatch(r"cuda:[0-9]+", args.device) is None:
        parser.error("--device must be a CUDA device such as cuda:0 or cuda:1")
    selection = dict(skip_gpu=args.skip_gpu, robot=args.robot, solutions=args.solutions,
                     include_clean_table=args.include_clean_table)
    if args.include_newton_cuda:
        selection["include_newton_cuda"] = True
    if args.device is not None:
        selection["device"] = args.device
    sys.exit(0 if run_final_check(**selection) else 1)


if __name__ == "__main__":
    main()
