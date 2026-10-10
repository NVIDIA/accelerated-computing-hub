# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Independent Newton worlds must complete real tasks, not template duplication."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"))
import benchmark_newton_batch as batch  # noqa: E402


@pytest.mark.parametrize("invalid", [
    {"scope": "replay"}, {"task": "stack"}, {"backend": "mujoco"}, {"worlds": 0}, {"worlds": True},
    {"frames": 300}, {"substeps": 1}, {"substeps": 10}, {"warmups": 0}, {"repeats": 0},
    {"cpu_threads": 0}, {"validation_workers": 0}, {"nconmax": -1}, {"njmax": .5}, {"max_trajectory_mib": float("nan")},
    {"backend": "newton_cuda", "device": "cpu"},
])
def test_invalid_or_shortened_batch_cannot_be_measured(invalid):
    with pytest.raises(ValueError):
        batch._configuration(invalid)


def test_memory_guard_keeps_full_2048_world_task():
    config = batch._configuration({"worlds": 2048})
    assert config["frames"] * config["substeps"] == 40000
    with pytest.raises(ValueError, match="cannot shorten"):
        batch._configuration({"worlds": 2048, "max_trajectory_mib": 64})


def test_protocol_signature_binds_world_count_but_not_device_backend():
    a = batch._configuration({"worlds": 16})
    tape, phases, _, cpu = batch._prepare(a)
    gpu_tape, gpu_phases, _, gpu = batch._prepare({**a, "backend": "newton_cuda"})
    assert cpu["comparison_signature"] == gpu["comparison_signature"]
    np.testing.assert_array_equal(tape, gpu_tape)
    np.testing.assert_array_equal(phases, gpu_phases)
    assert tape.dtype == np.float32 and tape.shape[0] == 2000
    assert cpu["physics_dt_seconds"] == .001 and cpu["physics_steps"] == 40000
    assert cpu["acceptance"]["protocol_version"] == 2
    _, _, _, other = batch._prepare({**a, "worlds": 32})
    assert other["comparison_signature"] != cpu["comparison_signature"]
    _, _, _, serial_validation = batch._prepare({**a, "validation_workers": 1})
    assert serial_validation["comparison_signature"] != cpu["comparison_signature"]


@pytest.mark.parametrize("robot", ("so101", "rebot"))
def test_real_cpu_pool_runs_every_world_and_resets_across_worlds_and_repeats(robot):
    # More worlds than workers proves reuse does not inherit a completed scene.
    row = batch.run_case({"robot": robot, "worlds": 3, "cpu_threads": 2, "warmups": 1, "repeats": 2})
    assert row["status"] == "passed", row.get("error")
    assert row["device_info"]["cpu_workers"] == 2
    assert row["host_validation"]["workers"] == 2
    assert row["host_validation"]["mode"] == "persistent_spawn_pool"
    assert row["timings"]["validation_setup_seconds"] > 0
    assert row["timings"]["validation_teardown_seconds"] > 0
    assert row["workload"]["host_validation_policy"]["requested_workers"] == 2
    assert row["device_info"]["cpu_worker_environment"]["CUDA_VISIBLE_DEVICES"] == ""
    episodes = row["warmup_samples"] + row["samples"]
    assert len(episodes) == 3
    for episode in episodes:
        assert episode["task_success_count"] == episode["task_total_count"] == 3
        assert episode["diagnostics"]["completed_worlds"] == 3
        assert {world["world"] for world in episode["validation"]["worlds"]} == {0, 1, 2}
        for world in episode["diagnostics"]["worlds"]:
            assert world["recorded_physics_steps"] == 40000
            assert world["final_time_seconds"] == pytest.approx(40)
            assert not world["warnings"]
        assert all(world["box_task"]["success"] and all(cube["grasped"] and cube["lifted"]
                   and cube["carried"] and cube["released"] and cube["detached_settled_frames"] >= 50
                   for cube in world["box_task"]["objects"].values())
                   for world in episode["validation"]["worlds"])
        assert all(world["clock"]["integration_count_passed"]
                   and world["clock"]["backend_clock_precision"] == "float64"
                   and world["clock"]["mismatched_frame_count"] == 0
                   for world in episode["validation"]["worlds"])
    options = row["workload"]["backend_settings"]
    assert options["active_backend"] == "native_mujoco"
    assert options["native_requested"]["timestep"] == .001
    assert options["uploaded_gpu_effective"] is None
    json.dumps(row, allow_nan=False)


def test_failed_full_warmup_never_yields_a_fast_measured_result(monkeypatch):
    original = batch._prepare

    def static_targets(config):
        tape, phases, spec, protocol = original(config)
        tape[:] = spec.home_ctrl
        return tape, phases, spec, protocol

    monkeypatch.setattr(batch, "_prepare", static_targets)
    row = batch.run_case({"worlds": 2, "cpu_threads": 1, "warmups": 1, "repeats": 1})
    assert row["status"] == "failed"
    assert row["samples"] == []
    assert len(row["warmup_samples"]) == 1
    assert row["warmup_samples"][0]["task_success_count"] == 0


def test_one_failed_world_is_not_hidden_by_successful_worlds():
    # Use measured physical histories from a real Newton CPU world, then corrupt
    # one world only. The common validator must reject the whole batch.
    config = batch._configuration({"worlds": 1})
    tape, phases, spec, _ = batch._prepare(config)
    world = batch._CPUWorld(config, tape)
    history = np.empty((2000, 61), dtype=np.float32)
    assert world.episode(history)["capacity_passed"]
    histories = np.stack([history, history.copy()])
    histories[1, :, 0] = 0
    result = batch.box_protocol.validate_observations(
        histories, spec, phases, clock_precision="float64", step_counts=[40000, 40000])
    assert not result["passed"]
    assert result["task_success_count"] == 1
    assert result["worlds"][0]["passed"] and not result["worlds"][1]["passed"]
    # Correct geometry and timestamps never substitute for a missing step.
    histories[1] = history
    result = batch.box_protocol.validate_observations(
        histories, spec, phases, clock_precision="float64", step_counts=[40000, 39999])
    assert not result["passed"] and result["task_success_count"] == 1
    assert not result["worlds"][1]["clock"]["integration_count_passed"]


def test_gpu_diagnostics_count_integrations_once_per_world_not_observation_forwards():
    """Run the CUDA diagnostic kernel on Warp CPU; no GPU physics claim."""
    import warp as wp
    wp.init()
    with wp.ScopedDevice("cpu"):
        worlds, bodies = 2, 3
        poses = wp.array([wp.transform_identity()] * (worlds * bodies), dtype=wp.transform)
        velocities = wp.zeros(worlds * bodies, dtype=wp.spatial_vector)
        bad = wp.zeros(worlds, dtype=int)
        contacts = wp.array([7], dtype=int)
        constraints = wp.array([3, 5], dtype=int)
        peak_contacts, peak_constraints = wp.zeros(1, dtype=int), wp.zeros(worlds, dtype=int)
        counts = wp.zeros(worlds, dtype=int)
        broadphase_pairs, peak_broadphase_pairs = wp.array([11], dtype=int), wp.zeros(1, dtype=int)
        common = [poses, velocities, bad, bodies, contacts, constraints,
                  peak_contacts, peak_constraints, counts]
        for _ in range(20):
            wp.launch(batch._diagnostics, (worlds, bodies),
                      inputs=[*common, 1, broadphase_pairs, peak_broadphase_pairs])
        np.testing.assert_array_equal(peak_broadphase_pairs.numpy(), [11])
        # Raw broadphase overflow in an observation-only forward must survive
        # even though no integration occurred to set a sticky overflow flag.
        broadphase_pairs.assign(np.array([129], dtype=np.int32))
        wp.launch(batch._diagnostics, (worlds, bodies),
                  inputs=[*common, 0, broadphase_pairs, peak_broadphase_pairs])
        np.testing.assert_array_equal(peak_broadphase_pairs.numpy(), [129])
        np.testing.assert_array_equal(counts.numpy(), [20, 20])
        np.testing.assert_array_equal(peak_constraints.numpy(), [3, 5])
        np.testing.assert_array_equal(bad.numpy(), [0, 0])
        counts.zero_()
        broadphase_pairs.assign(np.array([3], dtype=np.int32))
        wp.launch(batch._diagnostics, (worlds, bodies),
                  inputs=[*common, 0, broadphase_pairs, peak_broadphase_pairs])
        np.testing.assert_array_equal(peak_broadphase_pairs.numpy(), [129])
        np.testing.assert_array_equal(counts.numpy(), [0, 0])


def test_raw_broadphase_overflow_rejects_episode_even_without_sticky_flags(monkeypatch):
    """Exercise real acceptance logic using downloaded diagnostic fixtures."""
    from types import SimpleNamespace
    import mujoco_warp as mjw
    import warp as wp

    wp.init()
    with wp.ScopedDevice("cpu"):
        engine = batch._CUDABatch.__new__(batch._CUDABatch)
        engine.device, engine.mjw = wp.get_device("cpu"), mjw
        engine.reset = lambda: None
        engine.graph = object()
        monkeypatch.setattr(wp, "capture_launch", lambda graph: None)
        engine.output = wp.zeros((2, batch.FRAMES, batch.FIELDS), dtype=float)
        engine.data = SimpleNamespace(overflow=wp.zeros(2, dtype=int), naconmax=128, njmax=1024)
        engine.bad = wp.zeros(2, dtype=int)
        engine.cursor = wp.array([batch.FRAMES], dtype=int)
        engine.peak_contacts = wp.array([7], dtype=int)
        engine.peak_constraints = wp.array([3, 5], dtype=int)
        engine.step_counts = wp.array([40000, 40000], dtype=int)
        engine.peak_broadphase_pairs = wp.array([128], dtype=int)
        _, _, _, valid = engine.episode()
        assert valid["capacity_passed"]
        engine.peak_broadphase_pairs.assign(np.array([129], dtype=np.int32))
        _, _, _, invalid = engine.episode()
        assert invalid["overflow_flags_per_world"] == [0, 0]
        assert invalid["peak_contacts_all_worlds"] == 7
        assert invalid["peak_constraints_per_world"] == [3, 5]
        assert invalid["peak_broadphase_pairs_all_worlds"] == 129
        assert invalid["naconmax_all_worlds"] == 128
        assert not invalid["capacity_passed"]


def test_uploaded_options_report_effective_precision_defaults_without_changing_colliders():
    """Construct MJWarp arrays on CPU to inspect options, without a CUDA run."""
    import warp as wp
    wp.init()
    with wp.ScopedDevice("cpu"):
        config = batch._configuration({"backend": "newton_cuda", "worlds": 2})
        tape, _, _, _ = batch._prepare(config)
        _, solver, _, _, _, _, _, metadata = batch._build(config, 2, tape[0])
        settings = metadata["backend_settings"]
        native, uploaded = settings["native_requested"], settings["uploaded_gpu_effective"]
        assert native["timestep"] == .001
        assert uploaded["timestep"]["uniform_value"] == pytest.approx(.001)
        assert uploaded["tolerance"]["uniform_value"] == pytest.approx(max(native["tolerance"], 1e-6))
        assert uploaded["ls_tolerance"]["uniform_value"] == pytest.approx(native["ls_tolerance"])
        assert uploaded["tolerance"]["dtype"] == "float32"
        assert native["disableflags"] == uploaded["disableflags"]
        assert not native["disableflags"] & int(batch.mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        assert solver.mjw_model.opt.run_collision_detection
