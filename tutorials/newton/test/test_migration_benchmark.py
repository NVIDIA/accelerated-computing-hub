# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Reporting gates must never promote incomplete physics to a speedup."""

import copy
import json
from pathlib import Path
import sys

import pytest

PART3 = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"
sys.path.insert(0, str(PART3))
import migration_benchmark as benchmark  # noqa: E402


def completed_row(backend="mujoco", seconds=2.0, threads=1):
    row = {
        "case_id": f"replay-{backend}-w16-t{threads}", "status": "passed",
        "config": {"scope": "replay", "backend": backend, "robot": "so101", "worlds": 16,
                   "cpu_threads": threads, "frames": 600, "substeps": 10, "repeats": 3},
        "workload": {"comparison_signature": "identical-model-controls-and-initial-state"},
        "samples": [{"simulation_seconds": seconds, "output_transfer_seconds": 0.1,
                     "validation_seconds": 0.2, "task_success_count": 16, "task_total_count": 16,
                     "passed": True} for _ in range(3)],
    }
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


def test_cpu_only_both_scope_has_explicit_native_and_newton_baselines():
    args = benchmark.create_parser().parse_args(["--cpu-only", "--scope", "both", "--worlds", "1", "16"])
    cases = benchmark.case_plan(args)
    assert not any(case["backend"] in ("mjwarp", "newton_cuda") for case in cases)
    assert {(case["scope"], case["backend"]) for case in cases} == {
        ("replay", "mujoco"), ("workflow", "mujoco"), ("workflow", "newton_cpu")}
    assert sum(case["worlds"] == 1 and case["scope"] == "replay" for case in cases) == 1


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


def newton_row(backend="newton_cpu", seconds=2.0):
    row = completed_row(backend, seconds)
    row["case_id"] = f"workflow-{backend}-w1-t1"
    row["config"].update(scope="workflow", worlds=1, warmups=1)
    for sample in row["samples"]:
        sample.update(task_success_count=1, task_total_count=1)
    row["warmup_samples"] = [copy.deepcopy(row["samples"][0])]
    row["workload"] = {
        "task": "stack", "frames": 600, "physics_steps": 6000, "simulated_seconds": 12,
        "control_hz": 50, "physics_dt_seconds": .002, "controller": "online IK",
        "control_policy": "online", "rendering": False, "contact_detection": "MuJoCo",
        "output_policy": "payload history", "timed_scope": "controller and stepping",
        "observation_policy": "50 Hz", "asset_ref": "pinned-asset",
        "source_sha256": {"benchmark_workflows.py": "same-source"},
        "model_dimensions": {"nq": 20}, "solver_options": {"iterations": 100},
        "payload_observations": 600, "model_sha256": "same-model", "model_mjb_sha256": "same-model",
    }
    row["summary"] = benchmark.summarize_row(row)
    return row


def test_newton_comparison_is_explicit_and_separate_from_replay(tmp_path):
    cpu, gpu = newton_row(), newton_row("newton_cuda", 4)
    report = {"status": "passed", "configuration": {"robot": "so101"}, "cases": [cpu, gpu]}
    benchmark.write_outputs(tmp_path, report)
    saved = json.loads((tmp_path / "results.json").read_text())
    result, = saved["newton_workflow_comparisons"]
    assert result["compatible"]
    assert result["workflow_cost_ratio"] == .5
    assert result["checked_results_cost_ratio"] == pytest.approx(2.3 / 4.3)
    assert saved["comparisons"] == []
    assert "Newton CPU vs Newton CUDA" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("group,key,value", [
    ("config", "frames", 300), ("config", "warmups", 0), ("config", "worlds", 16),
    ("workload", "source_sha256", {"benchmark_workflows.py": "different-source"}),
    ("workload", "solver_options", {"iterations": 10}),
    ("workload", "timed_scope", "physics only"),
    ("workload", "source_sha256", {}),
])
def test_newton_mismatches_preserve_observations_without_ratio(group, key, value):
    cpu, gpu = newton_row(), newton_row("newton_cuda", 1)
    gpu[group][key] = value
    if key == "worlds":
        for sample in [*gpu["warmup_samples"], *gpu["samples"]]:
            sample.update(task_success_count=value, task_total_count=value)
    if key == "warmups":
        gpu["warmup_samples"] = []
    result, = benchmark.newton_workflow_comparisons([cpu, gpu])
    assert not result["compatible"]
    assert f"{group}.{key}" in result["mismatches"]
    assert "workflow_cost_ratio" not in result
    assert result["gpu_median_seconds"] == 1


@pytest.mark.parametrize("failure", ("partial", "failed", "native", "replay", "robot", "nan"))
def test_newton_does_not_pair_failed_or_different_scopes(failure):
    cpu, gpu = newton_row(), newton_row("newton_cuda", 1)
    if failure == "partial":
        gpu["samples"].pop()
    elif failure == "failed":
        gpu["status"] = "failed"
    elif failure == "native":
        cpu["config"]["backend"] = "mujoco"
    elif failure == "replay":
        gpu["config"]["scope"] = "replay"
    elif failure == "robot":
        gpu["config"]["robot"] = "rebot"
    elif failure == "nan":
        gpu["samples"][0]["simulation_seconds"] = float("nan")
    assert benchmark.newton_workflow_comparisons([cpu, gpu]) == []


def test_newton_missing_source_identity_does_not_produce_ratio():
    cpu, gpu = newton_row(), newton_row("newton_cuda", 1)
    del gpu["workload"]["source_sha256"]
    result, = benchmark.newton_workflow_comparisons([cpu, gpu])
    assert not result["compatible"]
    assert "workflow_cost_ratio" not in result


def test_newton_backend_models_can_differ_for_application_cost_ratio():
    cpu, gpu = newton_row(), newton_row("newton_cuda", 4)
    gpu["workload"].update(model_sha256="cuda-model", model_mjb_sha256="cuda-model")
    result, = benchmark.newton_workflow_comparisons([cpu, gpu])
    assert result["compatible"]
    assert result["workflow_cost_ratio"] == .5
    assert not result["compiled_models_identical"]
    assert result["cpu_model_sha256"] == "same-model"
    assert result["gpu_model_sha256"] == "cuda-model"


@pytest.mark.parametrize("value", (None, "", "inconsistent-alias"))
def test_newton_requires_recorded_consistent_model_identity(value):
    cpu, gpu = newton_row(), newton_row("newton_cuda", 1)
    gpu["workload"]["model_mjb_sha256"] = value
    result, = benchmark.newton_workflow_comparisons([cpu, gpu])
    assert not result["compatible"]
    assert "gpu.model_identity_missing_or_inconsistent" in result["mismatches"]
    assert "workflow_cost_ratio" not in result


def newton_batch_row(backend="newton_cpu", seconds=2.0):
    row = completed_row(backend, seconds, threads=8 if backend == "newton_cpu" else 1)
    row["case_id"] = row["case_id"].replace("replay-", "newton-batch-")
    row["config"].update(scope="newton-batch", task="box", frames=2000, substeps=20, warmups=1)
    row["warmup_samples"] = [copy.deepcopy(row["samples"][0])]
    row["workload"].update(task="two_cube_pick_place_into_box",frames=2000,physics_steps=40000,simulated_seconds=40.0,
                           physics_dt_seconds=.001, acceptance={"protocol_version": 2},
                           comparison_signature="same-newton-definition-tape-and-protocol",
                           model_mjb_sha256=f"{backend}-compiled-template")
    row["device_info"] = {"cpu_workers": 8 if backend == "newton_cpu" else 1}
    return row


def test_newton_batch_plan_covers_all_requested_environment_counts():
    worlds = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
    assert benchmark.PRESETS["workstation"] == worlds
    args = benchmark.create_parser().parse_args([
        "--scope", "newton-batch", "--cpu-threads", "32", "--worlds", *map(str, worlds)])
    cases = benchmark.case_plan(args)
    assert len(cases) == 18
    for count in worlds:
        pair = {case["backend"]: case for case in cases if case["worlds"] == count}
        assert set(pair) == {"newton_cpu", "newton_cuda"}
        assert pair["newton_cpu"]["cpu_threads"] == min(32, count)
        assert all(case["frames"] == 2000 and case["substeps"] == 20 and case["task"] == "box" for case in pair.values())


def test_newton_batch_memory_preflight_reports_2048_histories_and_budget():
    args = benchmark.create_parser().parse_args([
        "--scope", "newton-batch", "--worlds", "1", "2048", "--max-trajectory-mib", "8192"])
    report = benchmark.newton_batch_memory_preflight(args)
    assert report["within_trajectory_budget"]
    assert report["cases"][-1]["gpu_device_history_bytes"] == 2048 * 2000 * 61 * 4
    assert "solver allocations" in report["excludes"]
    args.max_trajectory_mib = 1
    assert not benchmark.newton_batch_memory_preflight(args)["within_trajectory_budget"]


def test_newton_batch_reports_cpu_processes_and_preserves_backend_models(tmp_path):
    cpu, gpu = newton_batch_row(), newton_batch_row("newton_cuda", 1)
    report = {"status": "passed", "configuration": {"robot": "so101"}, "cases": [cpu, gpu]}
    benchmark.write_outputs(tmp_path, report)
    saved = json.loads((tmp_path / "results.json").read_text())
    result, = saved["newton_batch_comparisons"]
    assert result["worlds"] == 16 and result["cpu_workers"] == 8
    assert result["cpu_parallelism"] == "persistent worker processes"
    assert result["batch_cost_ratio"] == 2
    assert not result["compiled_models_identical"]
    assert saved["comparisons"] == [] and saved["newton_workflow_comparisons"] == []
    assert "across environment counts" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("failure", ("worlds", "signature", "model", "partial", "failed-world", "scope", "warmups"))
def test_newton_batch_rejects_incomplete_or_incompatible_pair(failure):
    cpu, gpu = newton_batch_row(), newton_batch_row("newton_cuda", 1)
    if failure == "worlds":
        gpu["config"]["worlds"] = 32
        for sample in gpu["samples"]:
            sample.update(task_success_count=32, task_total_count=32)
    elif failure == "signature":
        gpu["workload"]["comparison_signature"] = "different-controls"
    elif failure == "model":
        del gpu["workload"]["model_mjb_sha256"]
    elif failure == "partial":
        gpu["samples"].pop()
    elif failure == "failed-world":
        gpu["samples"][0]["task_success_count"] = 15
    elif failure == "scope":
        gpu["config"]["scope"] = "workflow"
    else:
        gpu["config"]["warmups"] = 2
    assert benchmark.newton_batch_comparisons([cpu, gpu]) == []


@pytest.mark.parametrize("failure", ("missing", "failed", "partial-worlds", "wrong-count"))
def test_required_warmups_are_part_of_publication_eligibility(failure):
    row = newton_batch_row()
    if failure == "missing":
        del row["warmup_samples"]
    elif failure == "failed":
        row["warmup_samples"][0]["passed"] = False
    elif failure == "partial-worlds":
        row["warmup_samples"][0]["task_success_count"] -= 1
    else:
        row["warmup_samples"].append(copy.deepcopy(row["warmup_samples"][0]))
    assert not benchmark.summarize_row(row)["eligible"]
    assert benchmark.newton_batch_comparisons([row, newton_batch_row("newton_cuda", 1)]) == []


@pytest.mark.parametrize("field,value", (("task", "stack"), ("frames", 600),
                                         ("physics_steps", 6000), ("simulated_seconds", 12)))
def test_historical_stacking_pairs_never_enter_box_comparison(field, value):
    cpu, gpu = newton_batch_row(), newton_batch_row("newton_cuda", 1)
    for row in (cpu, gpu):
        row["workload"][field] = value
    assert benchmark.newton_batch_comparisons([cpu, gpu]) == []


@pytest.mark.parametrize("failure", ("old-timestep", "old-step-count", "old-protocol", "diagnostic"))
def test_box_v1_and_private_diagnostics_never_enter_v2_comparison(failure):
    cpu, gpu = newton_batch_row(), newton_batch_row("newton_cuda", 1)
    for row in (cpu, gpu):
        if failure == "old-timestep":
            row["workload"]["physics_dt_seconds"] = .002
        elif failure == "old-step-count":
            row["workload"]["physics_steps"] = 20000
        elif failure == "old-protocol":
            row["workload"]["acceptance"].pop("protocol_version")
        else:
            row["diagnostic_only"] = True
    assert benchmark.newton_batch_comparisons([cpu, gpu]) == []


def test_timeout_closes_worker_tree_before_next_case(tmp_path):
    import os
    import subprocess

    if os.name != "posix":
        pytest.skip("Benchmark process groups require POSIX")
    child_pid, cleanup = tmp_path / "child.pid", tmp_path / "cleanup"
    child = ("import os,signal,time;from pathlib import Path;"
             "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
             f"Path({str(child_pid)!r}).write_text(str(os.getpid()));time.sleep(60)")
    parent = ("import signal,subprocess,sys,time\nfrom pathlib import Path\n"
              "def terminate(signum,frame): raise SystemExit(128+signum)\n"
              "signal.signal(signal.SIGTERM,terminate)\n"
              f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
              "try: time.sleep(60)\n"
              f"finally: Path({str(cleanup)!r}).write_text('closed')\n")
    with (tmp_path / "worker.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            benchmark.run_isolated_worker([sys.executable, "-c", parent], environment=os.environ.copy(),
                                           log=log, timeout=1)
    assert cleanup.read_text() == "closed"
    pid = int(child_pid.read_text())
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    # A killed child can briefly remain an OS zombie before its reaper runs;
    # it cannot compute or contaminate the following timed case.
    assert not status.stdout.strip() or status.stdout.strip().startswith("Z")


def test_batch_cpu_and_gpu_share_requested_validation_worker_budget():
    args = benchmark.create_parser().parse_args(["--scope", "newton-batch", "--worlds", "1", "16", "--cpu-threads", "32", "--max-trajectory-mib", "8192"])
    cases = benchmark.case_plan(args)
    assert all(case["validation_workers"] == 32 for case in cases)
    assert all(case["cpu_threads"] == 1 for case in cases if case["backend"] == "newton_cuda")
    preflight = benchmark.newton_batch_memory_preflight(args)
    assert preflight["cases"][0]["validation_shared_history_bytes"] == 0
    assert preflight["cases"][1]["validation_shared_history_bytes"] == 16*2000*61*4
