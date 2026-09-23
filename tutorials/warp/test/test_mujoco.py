# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT

"""CPU task validation and CUDA-only MuJoCo Warp integration checks.

Run from the repository root with the tutorial environment:
    python -m pytest tutorials/warp/test/test_mujoco.py -v

The first task test downloads the pinned SO-101 Menagerie assets unless
MUJOCO_MENAGERIE_PATH or MUJOCO_MENAGERIE_CACHE already provides them.
"""

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1] / "notebooks" / "mujoco"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common():
    sys.path.insert(0, str(ROOT / "part1"))
    try:
        yield load_module(ROOT / "part1" / "pick_place_common.py", "tutorial_task")
    finally:
        sys.path.remove(str(ROOT / "part1"))


@pytest.fixture(scope="module")
def assets(common):
    from robots import get_robot
    from utils import resolve_menagerie_robot_path

    return resolve_menagerie_robot_path(get_robot("so101")).parent


def run_script(relative_path, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / relative_path), *map(str, args)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=900,
        check=False,
    )


def test_cpu_solution_completes_settled_stack(assets):
    result = run_script(
        "part1/solutions/so101_pick_place_solution.py",
        "--menagerie-path", assets, "--headless-steps", 600, "--test",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Stack validation passed" in result.stdout
    assert "dt=0.00200s" in result.stdout


def test_cpu_incomplete_task_fails(assets):
    result = run_script(
        "part1/solutions/so101_pick_place_solution.py",
        "--menagerie-path", assets, "--headless-steps", 1, "--test",
    )
    assert result.returncode != 0
    assert "Stack validation failed" in result.stderr


def test_relative_menagerie_path_builds_valid_assets_link(common, assets, tmp_path, monkeypatch):
    # Isolate generated files so an earlier absolute symlink cannot mask this bug.
    monkeypatch.setattr(common, "__file__", str(tmp_path / "pick_place_common.py"))
    monkeypatch.chdir(tmp_path)
    relative_assets = Path(os.path.relpath(assets, tmp_path))
    scene = common.resolve_pick_place_scene(relative_assets, explicit=True)
    assert (scene.parent / "assets").exists()
    model = common.load_pick_place_model(scene)
    assert model.nu == 6



def test_asset_link_follows_a_changed_explicit_checkout(common, tmp_path, monkeypatch):
    from robots import get_robot

    spec = get_robot("so101")
    monkeypatch.setattr(common, "__file__", str(tmp_path / "pick_place_common.py"))
    targets = []
    for name in ("first", "second"):
        root = tmp_path / name
        robot = root / spec.folder
        (robot / "assets").mkdir(parents=True)
        (robot / spec.robot_xml).write_text("<mujoco/>")
        scene = common.resolve_pick_place_scene(root, explicit=True, spec=spec)
        targets.append((scene.parent / "assets").resolve())
        assert targets[-1] == robot / "assets"
    assert targets[0] != targets[1]


def test_managed_cache_is_commit_specific_and_rejects_a_wrong_revision(common, tmp_path, monkeypatch):
    from robots import get_robot
    from utils import default_cache_dir, download_robot_sparse

    spec = get_robot("so101")
    monkeypatch.setenv("MUJOCO_MENAGERIE_CACHE", str(tmp_path))
    assert default_cache_dir(spec) == tmp_path / spec.cache_dirname / spec.menagerie_ref
    cache = tmp_path / "existing-cache"
    robot = cache / spec.folder
    robot.mkdir(parents=True)
    (robot / spec.robot_xml).write_text("<mujoco/>")
    subprocess.run(["git", "init", "-q", str(cache)], check=True)
    subprocess.run(["git", "-C", str(cache), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(cache), "-c", "user.name=Tutorial test", "-c",
         "user.email=tutorial-test@example.invalid", "-c", "commit.gpgsign=false",
         "commit", "-qm", "Test asset revision"],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(cache), "rev-parse", "HEAD"], text=True).strip()
    assert download_robot_sparse(replace(spec, menagerie_ref=revision), cache) == robot
    with pytest.raises(RuntimeError, match="not at the pinned commit"):
        download_robot_sparse(spec, cache)


def test_nonfinite_cpu_state_is_rejected(common):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    data.qpos[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        common.validate_cpu_state(data)


@pytest.fixture(scope="module")
def cuda():
    wp = pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("Requires an NVIDIA CUDA device; CPU task tests still run.")
    return wp


@pytest.fixture(scope="module")
def gpu_checks():
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    return load_module(ROOT / "part2" / "gpu_checks.py", "tutorial_gpu_checks")


def test_sticky_capacity_flag_is_rejected(gpu_checks):
    """A cleared current contact count must not hide an earlier overflow."""
    import mujoco_warp as mjw
    import warp as wp

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    with wp.ScopedDevice("cpu"):
        data = mjw.make_data(model, nworld=1, nconmax=8, njmax=8)
        data.overflow.assign(np.array([int(mjw.OverflowType.NEFC)], dtype=np.int32))
        data.nefc.zero_()
        with pytest.raises(RuntimeError, match="capacity overflow: NEFC"):
            gpu_checks.check_mjwarp_state(data)


def test_iteration_limit_is_reported_separately(gpu_checks):
    import mujoco_warp as mjw
    import warp as wp

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    with wp.ScopedDevice("cpu"):
        data = mjw.make_data(model, nworld=1, nconmax=8, njmax=8)
        data.overflow.assign(np.array([int(mjw.OverflowType.ITERATIONS)], dtype=np.int32))
        assert gpu_checks.check_mjwarp_state(data) == ["ITERATIONS"]


def test_cuda_solution_completes_settled_stack(cuda, assets):
    result = run_script(
        "part2/solutions/so101_mjwarp_solution.py",
        "--menagerie-path", assets, "--headless-steps", 600, "--test",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Stack validation passed" in result.stdout


def test_cuda_captured_benchmark_smoke(cuda, assets):
    result = run_script(
        "part2/solutions/so101_mjwarp_solution.py",
        "--menagerie-path", assets, "--benchmark", "--nworld", 2, "--steps", 3,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "timestep=0.002s" in result.stdout
    assert "graph=yes" in result.stdout
    assert "6 world-steps" in result.stdout


def test_cuda_real_constraint_overflow_is_rejected(cuda, gpu_checks):
    import mujoco_warp as mjw

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/>'
        '<body pos="0 0 0.02"><freejoint/><geom type="box" size="0.1 0.1 0.1"/>'
        '</body></worldbody></mujoco>'
    )
    with cuda.ScopedDevice("cuda:0"):
        device_model = mjw.put_model(model)
        data = mjw.make_data(model, nworld=1, nconmax=32, njmax=1)
        data.qpos.assign(np.asarray(model.qpos0[None, :], dtype=np.float32))
        mjw.step(device_model, data)
        with pytest.raises(RuntimeError, match="capacity overflow"):
            gpu_checks.check_mjwarp_state(data)
