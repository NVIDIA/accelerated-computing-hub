# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Reporting gates must never promote incomplete physics to a speedup."""

import copy
import json
from pathlib import Path
import sys

import pytest

PART3 = Path(__file__).resolve().parents[1] / "notebooks" / "newton" / "part3"
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
