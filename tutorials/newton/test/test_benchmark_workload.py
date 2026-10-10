# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Complete native tasks, repeat/reset checks and negative benchmark gates."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest


PART3 = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"
sys.path.insert(0, str(PART3))
import benchmark_workload as benchmark  # noqa: E402


@pytest.fixture(scope="module", params=("so101", "rebot"))
def complete_cpu_trajectory(request):
    config = benchmark._configuration(dict(backend="mujoco", robot=request.param, worlds=2,
                                          cpu_threads=2, repeats=2, warmups=1))
    prepared = benchmark._prepare(config)
    model, initial, initial_state, tape, phases, spec, _ = prepared
    replay = benchmark._CPUReplay(config, model, initial, initial_state, tape)
    try:
        first = replay.episode()[0].copy()
        second = replay.episode()[0].copy()
    finally:
        replay.close()
    return config, prepared, first, second


def test_full_task_succeeds_in_every_world_and_restarts(complete_cpu_trajectory):
    config, prepared, first, second = complete_cpu_trajectory
    model, _, _, _, phases, spec, _ = prepared
    np.testing.assert_array_equal(first, second)
    result = benchmark.validate_trajectory(first, model, spec, phases)
    assert result["passed"]
    assert result["task_success_count"] == config["worlds"]
    assert all(world["airborne_carry_m"] >= 0.05 for world in result["worlds"])


def test_serial_and_parallel_cpu_match(complete_cpu_trajectory):
    config, prepared, parallel, _ = complete_cpu_trajectory
    model, initial, initial_state, tape, _, _, _ = prepared
    replay = benchmark._CPUReplay(dict(config, cpu_threads=1), model, initial, initial_state, tape)
    try:
        serial = replay.episode()[0]
        np.testing.assert_array_equal(serial, parallel)
    finally:
        replay.close()


@pytest.mark.parametrize("failure", ("nan", "stalled_time", "no_lift", "bad_stack", "unsettled", "stack_on_floor", "interrupted_carry", "position_drift", "orientation_drift"))
def test_one_bad_world_rejects_the_batch(complete_cpu_trajectory, failure):
    _, prepared, trajectory, _ = complete_cpu_trajectory
    model, _, _, _, phases, spec, _ = prepared
    states = trajectory.copy()
    rq, rv = benchmark._cube_indices(model, "red_cube")
    if failure == "nan":
        states[1, 37, 2] = np.nan
    elif failure == "stalled_time":
        states[1, 37:, 0] = states[1, 36, 0]
    elif failure == "no_lift":
        # Preserve the final stack, but remove any lift/carry from its history.
        states[1, :5000, 1 + rq + 2] = spec.cube_center_z
    elif failure == "bad_stack":
        states[1, -500:, 1 + rq] += 0.1
    elif failure == "stack_on_floor":
        bq, _ = benchmark._cube_indices(model, "blue_cube")
        states[1, -500:, 1 + rq + 2] -= spec.table_top_z
        states[1, -500:, 1 + bq + 2] -= spec.table_top_z
    elif failure == "interrupted_carry":
        carry_frames = np.flatnonzero(phases == "above_blue")
        # First/last airborne points are far apart, separated by table contact.
        for frame in carry_frames[1:-1]:
            states[1, frame * 10:(frame + 1) * 10, 1 + rq + 2] = spec.cube_center_z
    elif failure == "position_drift":
        bq, _ = benchmark._cube_indices(model, "blue_cube")
        for qindex in (rq, bq):
            states[1, -500:, 1 + qindex] += np.linspace(0, 0.006, 500)
    elif failure == "orientation_drift":
        angle = np.linspace(0, np.deg2rad(4.0), 500)
        states[1, -500:, 1 + rq + 3:1 + rq + 7] = np.stack(
            (np.cos(angle / 2), np.zeros(500), np.zeros(500), np.sin(angle / 2)), axis=-1,
        )
    else:
        states[1, -500:, 1 + model.nq + rv] = 0.2
    result = benchmark.validate_trajectory(states, model, spec, phases)
    assert not result["passed"]
    assert result["task_success_count"] == 1
    assert result["worlds"][0]["passed"]
    assert not result["worlds"][1]["passed"]


def test_isolated_velocity_peak_is_reported_without_mislabeling_sustained_motion(complete_cpu_trajectory):
    _, prepared, trajectory, _ = complete_cpu_trajectory
    model, _, _, _, phases, spec, _ = prepared
    states = trajectory.copy()
    _, rv = benchmark._cube_indices(model, "red_cube")
    # A checker fixture: a single diagnostic velocity spike, unchanged raw
    # placement poses. Sustained motion is rejected in the negative tests.
    states[1, -1, 1 + model.nq + rv] = 0.08
    result = benchmark.validate_trajectory(states, model, spec, phases)
    assert result["passed"]
    assert result["worlds"][1]["settling_max_point_speed_bound_m_s"] >= 0.08
    assert result["worlds"][1]["settling_rms_point_speed_bound_m_s"] < 0.05


@pytest.mark.parametrize("change", ({"frames": 599}, {"substeps": 9}, {"warmups": 0},
                                    {"worlds": 0}, {"backend": "mjwarp", "device": "cpu"}))
def test_incomplete_or_invalid_cases_fail_before_running(change):
    row = benchmark.run_case(dict(backend="mujoco", robot="so101", **change)) if "backend" not in change else benchmark.run_case(change)
    assert row["status"] == "failed"
    assert not row["samples"]


def test_output_memory_limit_keeps_task_length():
    row = benchmark.run_case(dict(worlds=2, max_trajectory_mib=0.01))
    assert row["status"] == "failed"
    assert "reduce worlds" in row["diagnostics"]["error"]
    assert row["workload"]["physics_steps"] == 6000
    assert not row["samples"]


def test_case_has_serializable_samples_and_reproducible_inputs():
    config = dict(backend="mujoco", robot="so101", worlds=1, cpu_threads=1, repeats=2, warmups=1)
    first, second = benchmark.run_case(config), benchmark.run_case(config)
    assert first["status"] == second["status"] == "passed"
    json.dumps(first, allow_nan=False)
    assert len(first["samples"]) == 2
    assert len(first["warmup_samples"]) == 1
    for key in ("model_mjb_sha256", "control_tape_sha256", "initial_state_sha256"):
        assert first["workload"][key] == second["workload"][key]
    for sample in first["samples"]:
        assert sample["simulation_seconds"] > 0
        assert sample["output_transfer_seconds"] == 0
        assert sample["task_success_count"] == 1


def test_capacity_accepts_iteration_warnings_but_rejects_any_storage_failure():
    import mujoco_warp as mjw
    iteration_mask = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
    baseline = dict(flags=np.array([0, iteration_mask], dtype=np.int32), contacts=20,
                    constraints=np.array([20, 21], dtype=np.int32), iteration_mask=iteration_mask,
                    naconmax=128, njmax=300, cursor=6000, steps=6000)
    result = benchmark._capacity_diagnostics(**baseline)
    assert result["capacity_passed"]
    assert result["iteration_warning_flags_per_world"][1] == iteration_mask
    capacity_bit = next(int(member) for member in mjw.OverflowType if int(member) & ~iteration_mask)
    for changed in (dict(flags=np.array([0, capacity_bit], dtype=np.int32)),
                    dict(contacts=129), dict(constraints=np.array([20, 301])), dict(cursor=5999)):
        assert not benchmark._capacity_diagnostics(**dict(baseline, **changed))["capacity_passed"]


def test_device_recording_matches_native_fullphysics_layout(tmp_path, monkeypatch):
    import mujoco
    import warp as wp
    # Initialize before patching so teardown restores a valid cache path for
    # kernels compiled by later tests in the same process.
    wp.init()
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp"))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    expected = np.empty(nstate)
    mujoco.mj_getState(model, data, expected, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    with wp.ScopedDevice("cpu"):
        qpos, qvel, act = [wp.array(np.tile(value, (2, 1)), dtype=wp.float32)
                           for value in (data.qpos, data.qvel, data.act)]
        times = wp.array([data.time, data.time], dtype=wp.float32)
        cursor = wp.zeros(1, dtype=wp.int32)
        output = wp.empty((2, 1, nstate), dtype=wp.float32)
        wp.launch(benchmark._record_state, (2, nstate),
                  inputs=[qpos, qvel, act, times, cursor, model.nq, model.nv, output])
        np.testing.assert_allclose(output.numpy()[:, 0], np.tile(expected, (2, 1)), rtol=1e-6, atol=1e-8)


def test_cuda_complete_task_and_repeat_reset_when_available():
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA GPU unavailable; CPU tests are not CUDA validation")
    row = benchmark.run_case(dict(backend="mjwarp", robot="so101", worlds=2,
                                 repeats=2, warmups=1, device="cuda:0"))
    assert row["status"] == "passed", row
    assert all(sample["task_success_count"] == 2 for sample in row["samples"])
    assert all(sample["diagnostics"]["recorded_physics_steps"] == 6000 for sample in row["samples"])
    assert all(sample["diagnostics"]["capacity_passed"] for sample in row["samples"])
