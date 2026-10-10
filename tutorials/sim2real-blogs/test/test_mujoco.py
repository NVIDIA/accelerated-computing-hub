# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Actual single-world CLI checks in this tutorial's locked environment.

These checks may fetch pinned assets. They are functional checks, not benchmark
measurements. CPU physical gates also run in test_box_task.py; CUDA is skipped
when unavailable. The older Hub 3.12-only stack assertion API is not used.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / "notebooks/mujoco"


def run_box(relative, robot, report, *, frames=2000, device=None):
    command = [sys.executable, str(ROOT / relative), "--task", "box", "--robot", robot,
               "--sim-substeps", "20", "--headless-steps", str(frames), "--report", str(report)]
    if device:
        command += ["--device", device]
    return subprocess.run(command, capture_output=True, text=True, env=os.environ.copy(), timeout=900)


@pytest.mark.parametrize("robot", ["so101", "rebot"])
def test_cpu_incomplete_box_cannot_be_accepted(robot, tmp_path):
    report = tmp_path / "incomplete.json"
    result = run_box("part1/solutions/so101_pick_place_solution.py", robot, report, frames=1)
    assert result.returncode != 0, result.stdout + result.stderr
    assert report.is_file(), "Failed task must retain its physical report"
    assert json.loads(report.read_text())["success"] is False


@pytest.fixture(scope="module")
def cuda():
    wp = pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable; CPU results do not validate GPU execution")
    return "cuda:0"


@pytest.mark.parametrize("robot", ["so101", "rebot"])
def test_cuda_box_task_checks_both_cubes(robot, cuda, tmp_path):
    report = tmp_path / "box.json"
    result = run_box("part2/solutions/so101_mjwarp_solution.py", robot, report, device=cuda)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["success"] is True
    assert set(data["objects"]) == {"red_cube", "blue_cube"}
    assert data["tool_clearance_m"] >= .02
    for cube in data["objects"].values():
        assert cube["success"] is True
        assert cube["full_lift_frames"] >= 25
        assert cube["carry_distance_m"] >= .05
        assert cube["release_open_fraction"] >= .8
        assert cube["detached_settled_frames"] >= 50
