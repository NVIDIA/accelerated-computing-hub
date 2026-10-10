# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Reporting gates must never promote incomplete physics to a speedup."""

import copy
import json
from pathlib import Path
import sys

import pytest

# Load under an article-specific name without leaking shared tutorial aliases.
import importlib.util
PART2 = Path(__file__).resolve().parents[1] / "notebooks" / "mujoco" / "part2"
_aliases = ("pick_place_common", "robots", "utils")
_saved = {key: sys.modules.pop(key) for key in _aliases if key in sys.modules}
sys.path.insert(0, str(PART2))
try:
    _spec = importlib.util.spec_from_file_location("article2_migration_benchmark", PART2 / "migration_benchmark.py")
    benchmark = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = benchmark
    _spec.loader.exec_module(benchmark)
finally:
    sys.path.remove(str(PART2))
    for _key in _aliases:
        sys.modules.pop(_key, None)
    sys.modules.update(_saved)


def completed_row(backend="mujoco", seconds=2.0, threads=1):
    row = {
        "case_id": f"replay-{backend}-w16-t{threads}", "status": "passed",
        "config": {"scope": "replay", "backend": backend, "robot": "so101", "worlds": 16,
                   "cpu_threads": threads, "task": "stack", "frames": 600, "substeps": 10, "repeats": 3, "warmups": 1},
        "workload": {"comparison_signature": "identical-model-controls-and-initial-state"},
        "samples": [{"simulation_seconds": seconds, "output_transfer_seconds": 0.1,
                     "validation_seconds": 0.2, "task_success_count": 16, "task_total_count": 16,
                     "passed": True} for _ in range(3)],
    }
    row["warmup_samples"] = copy.deepcopy(row["samples"][:1])
    row["summary"] = benchmark.summarize_row(row)
    return row


@pytest.mark.parametrize("failure", ("partial", "failed-world", "failed-status", "nan", "zero", "negative-transfer", "missing-validation"))
def test_invalid_repetition_never_produces_speedup(failure):
    row = completed_row("mjwarp", 0.01)
    if failure == "partial":
        row["samples"].pop()
    elif failure == "failed-world":
        row["samples"][0]["task_success_count"] = 15
    elif failure == "failed-status":
        row["status"] = "failed"
    elif failure == "nan":
        row["samples"][0]["simulation_seconds"] = float("nan")
    elif failure == "zero":
        row["samples"][0]["simulation_seconds"] = 0
    elif failure == "negative-transfer":
        row["samples"][0]["output_transfer_seconds"] = -1
    else:
        del row["samples"][0]["validation_seconds"]
    row["summary"] = benchmark.summarize_row(row)
    assert not row["summary"]["eligible"]
    assert "median_seconds" not in row["summary"]
    assert benchmark.replay_comparisons([completed_row(), row]) == []


@pytest.mark.parametrize("mismatch", ("signature", "missing-signature", "worlds", "robot", "workflow"))
def test_different_workloads_never_produce_ratio(mismatch):
    gpu = completed_row("mjwarp", 1)
    if mismatch == "signature":
        gpu["workload"]["comparison_signature"] = "changed-physics"
    elif mismatch == "missing-signature":
        gpu["workload"].clear()
    elif mismatch == "workflow":
        gpu["config"]["scope"] = "workflow"
    elif mismatch == "robot":
        gpu["config"]["robot"] = "rebot"
    else:
        gpu["config"]["worlds"] = 1
    assert benchmark.replay_comparisons([completed_row(), gpu]) == []


def test_compare_fastest_measured_cpu_pool_and_actual_threads():
    serial, pool, gpu = completed_row(seconds=8), completed_row(seconds=2, threads=8), completed_row("mjwarp", 1)
    pool["device_info"] = {"cpu_threads": 4}  # Container quota may cap the requested count.
    result, = benchmark.replay_comparisons([serial, pool, gpu])
    assert result["simulation_speedup"] == 2
    assert result["cpu_threads"] == 4
    assert result["checked_results_speedup"] == pytest.approx(2.3 / 1.3)
    assert result["cpu_case"] == pool["case_id"]


def test_metrics_use_successful_episodes_and_all_physics_steps():
    stats = completed_row(seconds=2)["summary"]
    assert stats["successful_tasks_per_second"] == 8
    assert stats["world_steps_per_second"] == 48000
    assert stats["aggregate_real_time_factor"] == 96


def test_cpu_only_reports_no_gpu_claim_and_exports_raw_samples(tmp_path):
    row = completed_row()
    report = {"status": "passed", "configuration": {"robot": "so101"}, "cases": [row]}
    benchmark.write_outputs(tmp_path, report)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["cases"][0]["samples"] == row["samples"]
    assert saved["comparisons"] == []
    assert "no speedup or crossover is reported" in (tmp_path / "summary.md").read_text()
    assert "48000" in (tmp_path / "summary.csv").read_text()


def test_stale_case_directory_is_rejected_before_launch(tmp_path, monkeypatch):
    (tmp_path / "replay-mujoco-w1-t1.config.json").write_text("{}")
    monkeypatch.setattr(benchmark, "hardware_metadata", lambda: pytest.fail("must reject before probing"))
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--cpu-only", "--output-dir", str(tmp_path)])
    assert error.value.code == 2



def test_gpu_process_report_excludes_paths_and_arguments():
    assert benchmark.executable_name('/usr/bin/chrome --profile-directory="Private Work" --token=secret') == "chrome"
    assert benchmark.executable_name('"/opt/Example App/bin/runner" --secret value') == "runner"


def test_incomplete_report_still_preserves_successful_cpu_row(tmp_path):
    cpu = completed_row()
    missing = copy.deepcopy(cpu)
    missing.update(case_id="replay-mjwarp-w16-t1", status="unavailable", samples=[])
    missing["config"]["backend"] = "mjwarp"
    missing["summary"] = benchmark.summarize_row(missing)
    benchmark.write_outputs(tmp_path, {"status": "incomplete_or_failed", "configuration": {"robot": "so101"},
                                      "cases": [cpu, missing]})
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["cases"][0]["summary"]["eligible"]
    assert not report["cases"][1]["summary"]["eligible"]
    assert report["comparisons"] == []


@pytest.mark.parametrize("baseline,counts", [("pool", {1, 16, 32}), ("serial", {1}), ("both", {1, 16, 32})])
def test_complete_requested_sweep_has_cpu_and_gpu_for_every_world_count(baseline, counts):
    args = benchmark.create_parser().parse_args(["--preset", "workstation", "--cpu-baseline", baseline,
                                               "--cpu-threads", "32"])
    args.worlds = benchmark.PRESETS[args.preset]
    assert args.worlds == [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
    cases = benchmark.case_plan(args)
    assert {case["cpu_threads"] for case in cases if case["backend"] == "mujoco"} == counts
    for worlds in args.worlds:
        assert {case["backend"] for case in cases if case["worlds"] == worlds} == {"mujoco", "mjwarp"}
    assert len(cases) == (26 if baseline == "both" else 18)
    assert all(case["validation_workers"] == 32 for case in cases)
    assert all(case["frames"] * case["substeps"] == 40000 for case in cases)


def test_cpu_only_uses_native_baseline_without_newton():
    args = benchmark.create_parser().parse_args(["--cpu-only", "--worlds", "1", "16"])
    assert {case["backend"] for case in benchmark.case_plan(args)} == {"mujoco"}
    assert "newton" not in benchmark.PACKAGES


def test_memory_preflight_accounts_for_full_cpu_gpu_history(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout='{"nstate": 40, "nu": 6}', stderr=""))
    args = benchmark.create_parser().parse_args(["--task", "stack", "--worlds", "2048", "--max-trajectory-mib", "8192"])
    result = benchmark.memory_preflight(args, {})
    row, = result["cases"]
    assert row["cpu_trajectory_mib"] == 2048 * 6000 * 40 * 8 / 1024**2
    assert row["gpu_trajectory_mib"] == row["cpu_trajectory_mib"] / 2
    assert row["within_trajectory_limit"]
    args.max_trajectory_mib = 1024
    assert not benchmark.memory_preflight(args, {})["cases"][0]["within_trajectory_limit"]


@pytest.mark.parametrize("failure", ("missing", "failed", "partial", "bad-count", "nan"))
def test_warmup_evidence_is_required(failure):
    row = completed_row()
    if failure == "missing":
        del row["warmup_samples"]
    elif failure == "failed":
        row["warmup_samples"][0]["passed"] = False
    elif failure == "partial":
        row["config"]["warmups"] = 2
    elif failure == "bad-count":
        row["warmup_samples"][0]["task_success_count"] = 15
    else:
        row["warmup_samples"][0]["simulation_seconds"] = float("nan")
    assert not benchmark.summarize_row(row)["eligible"]


def test_box_defaults_full_task_memory_and_separate_legacy_option():
    args = benchmark.create_parser().parse_args([])
    assert args.task == "box" and args.preset == "workstation"
    assert args.repeats == 5 and args.cpu_threads == 32
    args.worlds = benchmark.PRESETS[args.preset]
    args.max_trajectory_mib = 8192
    cases = benchmark.case_plan(args)
    assert len(cases) == 18
    assert all(row["task"] == "box" and row["frames"] == 2000 for row in cases)
    memory = benchmark.memory_preflight(args, {})
    assert memory["observation_fields"] == 61
    assert memory["cases"][-1]["cpu_trajectory_mib"] == 2048*2000*61*4/1024**2
    args.task = "stack"
    legacy = benchmark.case_plan(args)
    assert all(row["frames"] == 600 for row in legacy)
    assert not {row["case_id"] for row in cases} & {row["case_id"] for row in legacy}


def test_box_pair_requires_full_box_protocol_even_if_signature_matches():
    cpu, gpu = completed_row(), completed_row("mjwarp", 1)
    for row in (cpu, gpu):
        row["config"].update(task="box", frames=2000, substeps=20)
    assert benchmark.replay_comparisons([cpu, gpu]) == []
    for row in (cpu, gpu):
        row["workload"].update(task="two_cube_pick_place_into_box", frames=2000,
                               physics_steps=40000, simulated_seconds=40., substeps=20, dt=.001, protocol_version=2)
    assert benchmark.replay_comparisons([cpu, gpu])[0]["task"] == "box"
    cpu["config"]["task"] = "stack"
    assert benchmark.replay_comparisons([cpu, gpu]) == []


def test_timeout_terminates_pool_descendants_before_next_case(tmp_path):
    import os
    import signal
    import subprocess
    import time
    if os.name != "posix":
        pytest.skip("POSIX isolated worker groups")
    heartbeat = tmp_path / "heartbeat"
    script = tmp_path / "worker.py"
    script.write_text("import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        + repr("import time; from pathlib import Path; p=Path(" + repr(str(heartbeat)) + "); "
               "exec('while True:\\n p.write_text(str(time.time()))\\n time.sleep(.05)')") + "])\n"
        "time.sleep(60)\n")
    with (tmp_path / "log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            benchmark.run_isolated_worker([sys.executable, str(script)], environment=os.environ.copy(),
                                          log=log, timeout=.5)
    assert heartbeat.exists()
    first = heartbeat.read_text()
    time.sleep(.2)
    assert heartbeat.read_text() == first


@pytest.mark.parametrize("location", ("row", "workload"))
@pytest.mark.parametrize("flag,value", [("diagnostic_only", True), ("publication_eligible", False)])
def test_diagnostic_receipts_never_become_performance_results(location, flag, value):
    row = completed_row()
    target = row if location == "row" else row["workload"]
    target[flag] = value
    assert not benchmark.summarize_row(row)["eligible"]


@pytest.mark.parametrize("field,value", [("physics_steps",20000), ("dt",.002), ("substeps",10), ("protocol_version",1)])
def test_box_v1_protocol_is_never_paired_as_v2(field, value):
    cpu, gpu = completed_row(), completed_row("mjwarp", 1)
    for row in (cpu, gpu):
        row["config"].update(task="box",frames=2000,substeps=20)
        row["workload"].update(task="two_cube_pick_place_into_box",frames=2000,physics_steps=40000,
                               simulated_seconds=40.,substeps=20,dt=.001,protocol_version=2)
        row["workload"][field] = value
    assert benchmark.replay_comparisons([cpu,gpu]) == []


def test_shared_validation_memory_is_in_preflight_for_both_devices():
    args = benchmark.create_parser().parse_args(["--worlds", "1", "16", "--cpu-threads", "32", "--max-trajectory-mib", "8192"])
    report = benchmark.memory_preflight(args, {})
    assert report["cases"][0]["validation_shared_history_mib"] == 0
    assert report["cases"][1]["validation_shared_history_mib"] == 16*2000*61*4/1024**2
