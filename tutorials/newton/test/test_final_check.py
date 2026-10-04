"""Final gate tests: no simulator or CUDA required for report validation."""
from __future__ import annotations

import contextlib
import copy
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from clean_table_report_fixture import make_gripper_report

PART3 = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"
spec = importlib.util.spec_from_file_location("blog3_final_check", PART3 / "final_check.py")
assert spec is not None and spec.loader is not None
final_check = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = final_check
spec.loader.exec_module(final_check)


class CheckSelectionTests(unittest.TestCase):
    def test_reference_mode_selects_solutions_without_changing_students(self):
        students = final_check.checks_for("so101")
        references = final_check.checks_for("so101", solutions=True)
        self.assertEqual(len(students), 3)
        self.assertEqual(len(references), 3)
        for student, reference in zip(students, references):
            self.assertNotEqual(student.script, reference.script)
            self.assertEqual(reference.script.parent.name, "solutions")
            self.assertEqual(reference.script.stem, student.script.stem + "_solution")
            self.assertEqual(student.args, reference.args)

    def test_newton_cuda_checks_cover_both_contacts_on_the_selected_device(self):
        checks = final_check.checks_for("rebot", solutions=True, include_newton_cuda=True,
                                       include_clean_table=True, device="cuda:1")
        self.assertEqual(len(checks), 6)
        self.assertEqual(checks[1].warp_device, "cuda:1")
        cpu = checks[2]
        self.assertEqual(cpu.args[cpu.args.index("--device") + 1], "cpu")
        for check, contacts in zip(checks[3:5], ("Newton", "MuJoCo")):
            with self.subTest(contacts=contacts):
                self.assertTrue(check.requires_cuda)
                self.assertEqual(check.device, "cuda:1")
                self.assertEqual(check.args[check.args.index("--device") + 1], "cuda:1")
                self.assertEqual("--use-mujoco-contacts" in check.args, contacts == "MuJoCo")
                self.assertEqual(check.script, PART3 / "solutions" / "so101_newton_solution.py")
                self.assertEqual(check.physics, f"Physics: MuJoCo Warp (CUDA); contacts: {contacts};")
        self.assertEqual(checks[-1].device, "cuda:1")
        self.assertEqual(checks[-1].args[-2:], ["--device", "cuda:1"])

    def test_selected_device_must_be_an_explicit_cuda_ordinal(self):
        for device in ("cpu", "cuda", "cuda:-1", "cuda:one", "cuda:1 extra"):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ValueError, "use cuda:N"):
                    final_check.checks_for("so101", device=device)
                with self.assertRaisesRegex(ValueError, "use cuda:N"):
                    final_check.run_final_check(skip_gpu=True, device=device)


class StackGateTests(unittest.TestCase):
    def run_script(self, source, *, timeout: float = 10):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "probe.py"
            script.write_text(source)
            return final_check.run_check(final_check.Check("probe", script, []), timeout=timeout)

    def test_stack_gate_is_strict_finite_and_uses_the_final_report(self):
        for report, expected in [
            ("xy_err=0.015 m dz=0.035 m", True),
            ("xy_err=1.0e-3 dz=5.5e-2", True),
            ("xy_err=0.020 dz=0.044", False),
            ("xy_err=0.001 dz=0.031", False),
            ("xy_err=-0.001 dz=0.044", False),
            ("xy_err=nan dz=0.044", False),
            ("xy_err=0.001 dz=inf", False),
            ("xy_err=bad dz=0.044", False),
            ("xy_err=0.001 dz=0.044\nxy_err=0.1 dz=0", False),
            ("No metric", False),
        ]:
            with self.subTest(report=report):
                ok, detail = self.run_script(f"print({report!r})")
                self.assertEqual(ok, expected, detail)

    def test_nonzero_exit_cannot_pass_even_with_good_metrics(self):
        ok, detail = self.run_script("print('xy_err=0.001 dz=0.044'); raise RuntimeError('broken')")
        self.assertFalse(ok)
        self.assertIn("broken", detail)

    def test_timeout_fails(self):
        ok, detail = self.run_script("import time; time.sleep(2)", timeout=0.05)
        self.assertFalse(ok)
        self.assertIn("timed out", detail)

    def test_cuda_stack_gate_requires_the_selected_backend_and_contact_path(self):
        check = final_check.checks_for("so101", include_newton_cuda=True)[-1]
        with tempfile.TemporaryDirectory() as directory:
            check.script = Path(directory) / "probe.py"
            check.args = []
            for backend, expected in (
                ("Physics: MuJoCo Warp (CUDA); contacts: MuJoCo; dt=0.002 s", True),
                ("Physics: native MuJoCo-C (CPU); contacts: MuJoCo; dt=0.002 s", False),
                ("Physics: MuJoCo Warp (CUDA); contacts: Newton; dt=0.002 s", False),
                ("", False),
            ):
                with self.subTest(backend=backend):
                    check.script.write_text(f"print({backend!r}); print('xy_err=0.001 dz=0.044')")
                    ok, detail = final_check.run_check(check)
                    self.assertEqual(ok, expected, detail)

    def test_legacy_mjwarp_child_selects_device_and_preserves_script_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            # A protocol fixture proves child-process selection without CUDA.
            (path / "warp.py").write_text(
                "def init(): pass\n"
                "def set_device(value):\n    global device\n    device = value\n"
            )
            script = path / "probe.py"
            script.write_text(
                "import sys, warp\n"
                "assert warp.device == 'cuda:1'\n"
                "assert sys.argv == ['probe.py', '--robot', 'so101']\n"
                "print('xy_err=0.001 dz=0.044')\n"
            )
            check = final_check.Check("selected GPU", script, ["--robot", "so101"], warp_device="cuda:1")
            ok, detail = final_check.run_check(check)
            self.assertTrue(ok, detail)


class CleanTableGateTests(unittest.TestCase):
    def test_clean_table_is_opt_in_and_has_long_timeout(self):
        normal = final_check.checks_for("rebot", solutions=True)
        extended = final_check.checks_for("rebot", solutions=True, include_clean_table=True)
        self.assertEqual(len(extended), len(normal) + 1)
        check = extended[-1]
        self.assertEqual(check.script, PART3 / "solutions" / "clean_the_table_solution.py")
        artifacts = PART3 / ".generated" / "final_check"
        self.assertEqual(check.args, ["--robot", "rebot", "--viewer", "null", "--test",
                                     "--report", str(artifacts / "reference_clean_rebot.json"),
                                     "--record", str(artifacts / "reference_clean_rebot.npz")])
        self.assertGreaterEqual(check.timeout, 1200)

    def test_only_clean_checks_record_separate_student_and_reference_artifacts(self):
        paths = set()
        for robot in ("so101", "rebot"):
            for solutions, mode in ((False, "student"), (True, "reference")):
                checks = final_check.checks_for(robot, solutions=solutions, include_clean_table=True)
                self.assertTrue(all("--report" not in check.args and "--record" not in check.args
                                    for check in checks[:-1]))
                clean = checks[-1]
                for flag, suffix in (("--report", ".json"), ("--record", ".npz")):
                    self.assertIn(flag, clean.args)
                    path = Path(clean.args[clean.args.index(flag) + 1])
                    self.assertEqual(path, PART3 / ".generated" / "final_check" / f"{mode}_clean_{robot}{suffix}")
                    self.assertNotIn(path, paths)
                    paths.add(path)

    def test_explicit_gpu_artifacts_do_not_overwrite_other_device_runs(self):
        for device in ("cuda:0", "cuda:1"):
            with self.subTest(device=device):
                clean = final_check.checks_for("so101", solutions=True, include_clean_table=True,
                                               device=device)[-1]
                for flag, suffix in (("--report", ".json"), ("--record", ".npz")):
                    path = Path(clean.args[clean.args.index(flag) + 1])
                    self.assertEqual(path.name, f"reference_clean_so101_{device.replace(':', '')}{suffix}")

    def test_clean_table_result_is_fail_closed(self):
        # Deliberate protocol fixture, not a claimed physics outcome.
        valid = make_gripper_report()
        variants = [(valid, True)]
        for field, value in (("success", False), ("success", "true"), ("phase", "sweep"),
                             ("frames", 0), ("frames", True), ("robot", "rebot"),
                             ("device", ""), ("device", "not-a-device"), ("objects", {}),
                             ("max_coupling_input_force_norm", 0.0), ("max_coupling_input_force_norm", True),
                             ("max_soft_contacts", 0), ("max_soft_contacts", True),
                             ("tool_withdrawn", False), ("tool_withdrawn", 1), ("tool_withdrawn", "true"),
                             ("tool_clearance_m", -0.01), ("tool_clearance_m", 0.019999),
                             ("tool_clearance_m", True), ("tool_clearance_m", "0.03"),
                             ("tool_clearance_m", None), ("tool_clearance_m", float("inf"))):
            modified = copy.deepcopy(valid)
            modified[field] = value
            variants.append((modified, False))
        for field, value in (("inside", False), ("inside", 1), ("settled_frames", 49),
                             ("settled_frames", True), ("settled_frames", 651),
                             ("max_point_speed", 0.04), ("max_point_speed", -0.01), ("max_point_speed", None)):
            modified = copy.deepcopy(valid)
            modified["objects"]["cable"][field] = value
            variants.append((modified, False))
        for field in valid:
            modified = copy.deepcopy(valid)
            del modified[field]
            variants.append((modified, False))
        modified = copy.deepcopy(valid)
        del modified["objects"]["shirt"]
        variants.append((modified, False))
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "protocol_fixture.py"
            check = final_check.checks_for("so101", include_clean_table=True)[-1]
            check.script = script
            for report, expected in variants:
                with self.subTest(report=report):
                    output = "CLEAN_TABLE_RESULT " + json.dumps(report)
                    script.write_text(f"print({output!r})")
                    ok, detail = final_check.run_check(check)
                    self.assertEqual(ok, expected, detail)
            for output in ("", "CLEAN_TABLE_RESULT {}", "CLEAN_TABLE_RESULT not-json",
                           "CLEAN_TABLE_RESULT " + json.dumps(valid) + "\nCLEAN_TABLE_RESULT bad",
                           "CLEAN_TABLE_RESULT " + json.dumps(valid) + "\nCLEAN_TABLE_RESULT",
                           "CLEAN_TABLE_RESULT " + json.dumps({**valid, "extra_metric": float("nan")})):
                with self.subTest(output=output):
                    script.write_text(f"print({output!r})")
                    self.assertFalse(final_check.run_check(check)[0])
            script.write_text(f"print({'CLEAN_TABLE_RESULT ' + json.dumps(valid)!r}); raise RuntimeError('failed')")
            self.assertFalse(final_check.run_check(check)[0])

    def test_clean_table_cannot_pass_on_an_unrequested_device(self):
        report = make_gripper_report()
        for actual, requested, expected in (("cpu", "cpu", True), ("cuda:1", "cuda:1", True),
                                             ("cpu", "cuda:1", False), ("cuda:0", "cuda:1", False),
                                             ("cuda:1", "cpu", False)):
            with self.subTest(actual=actual, requested=requested):
                report["device"] = actual
                output = "CLEAN_TABLE_RESULT " + json.dumps(report)
                ok, detail = final_check.validate_clean_table(output, robot="so101", device=requested)
                self.assertEqual(ok, expected, detail)

    def test_gripper_gate_rejects_legacy_sweeps_and_containment_without_manipulation(self):
        report = make_gripper_report()
        report["task"] = "physical_sweep_into_bin"
        self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])
        report = make_gripper_report()
        report["objects"] = {
            name: {key: obj[key] for key in ("inside", "settled_frames", "max_point_speed",
                                            "bounds_min", "bounds_max")}
            for name, obj in report["objects"].items()
        }
        # A full bin and invented aggregate feedback cannot replace an object's
        # measured grasp/lift/carry/release history, even with the new task label.
        self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])

    def test_every_payload_needs_a_sustained_loaded_lift_carry_and_detached_release(self):
        bad = (
            ("max_loaded_bilateral_frames", 9), ("full_lift_frames", 24),
            ("carry_frames", 9), ("detached_settled_frames", 49),
            ("full_lift_frames", True), ("min_lift_clearance_m", 0.009),
            ("min_jaw_normal_force_N", 0.0), ("min_jaw_normal_force_N", True),
            ("dropped_before_release", True), ("release_commanded", False),
            ("release_open_fraction", 0.79), ("release_open_fraction", True),
            ("jaw_contacts", [False, True]), ("jaw_contacts", [0, 1]),
        )
        for name in final_check.CLEAN_TABLE_OBJECTS:
            for field, value in bad:
                with self.subTest(payload=name, field=field, value=value):
                    report = make_gripper_report()
                    report["objects"][name][field] = value
                    ok, detail = final_check.validate_clean_table_report(report, robot="so101")
                    self.assertFalse(ok, detail)
        report = make_gripper_report()
        for obj in report["objects"].values():
            obj["jaw_contacts"] = [False, False]
        self.assertTrue(final_check.validate_clean_table_report(report, robot="so101")[0])

    def test_measured_bounds_and_airborne_distance_override_success_flags(self):
        variants = (
            {"bounds_max": [0.22, 0.02, 0.05]},  # trailing material outside bin
            {"carry_bounds_max": [0.22, 0.02, 0.18]},  # released before fully over bin
            {"carry_bounds_min": [0.08, -0.02, 0.1]},  # dragging on the table
            {"carry_distance_m": 0.5},  # claimed distance disagrees with measured centers
            {"carry_center": [0.1, 0.0, 0.3]},  # arrival center outside measured envelope
            {"carry_start_center": [0.1, -0.02, 0.16], "carry_distance_m": 0.02},
            {"grasp_center": [0.1, 0.0, 0.12]},  # object already above receiving area
        )
        for changed in variants:
            with self.subTest(changed=changed):
                report = make_gripper_report()
                report["objects"]["shirt"].update(changed)
                self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])

    def test_event_order_and_elapsed_time_must_support_the_measured_windows(self):
        for changed in (
            {"lift_time": 0.4}, {"carry_time": 0.9}, {"release_command_time": 1.4},
            {"release_time": 1.6}, {"release_time": 13.01}, {"grasp_time": True},
            {"grasp_time": 0.9, "lift_time": 1.0, "carry_time": 1.1},
            {"release_time": 12.5},  # not enough time for one second of detached settling
        ):
            with self.subTest(changed=changed):
                report = make_gripper_report()
                report["objects"]["red_cube"].update(changed)
                self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])
        report = make_gripper_report()
        for field in ("grasp_time", "lift_time", "carry_time", "release_command_time", "release_time"):
            report["objects"]["blue_cube"][field] = report["objects"]["red_cube"][field]
        self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])

    def test_saved_reports_reject_capacity_errors_nonfinite_values_and_unsupported_robots(self):
        for flags in (0, 512, 1024, 1536):
            report = make_gripper_report()
            report["mujoco_warp_overflow_flags"] = flags
            self.assertTrue(final_check.validate_clean_table_report(report, robot="so101")[0])
        for flags in (None, True, -1, 1, 1537, 2048):
            report = make_gripper_report()
            report["mujoco_warp_overflow_flags"] = flags
            self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])
        report = make_gripper_report("unrecognized")
        self.assertFalse(final_check.validate_clean_table_report(report, robot="unrecognized")[0])
        report = make_gripper_report()
        report["extra"] = float("nan")
        self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])
        for key, value in (("finite", False), ("failure", "payload fell before release")):
            report = make_gripper_report()
            report[key] = value
            self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])
        report = make_gripper_report()
        report["frames"] = 10 ** 400
        self.assertFalse(final_check.validate_clean_table_report(report, robot="so101")[0])


class FinalSummaryTests(unittest.TestCase):
    def test_skips_are_reported_separately_and_never_count_as_passes(self):
        checks = [final_check.Check("CPU", Path("cpu.py"), []),
                  final_check.Check("GPU", Path("gpu.py"), [], requires_cuda=True)]
        log = io.StringIO()
        with mock.patch.object(final_check, "checks_for", return_value=checks) as selection, \
             mock.patch.object(final_check, "cuda_available") as cuda, \
             mock.patch.object(final_check, "run_check", return_value=(True, "fixture passed")) as run, \
             contextlib.redirect_stdout(log):
            result = final_check.run_final_check(skip_gpu=True, robot="both", solutions=True, include_clean_table=True)
        self.assertTrue(result)
        cuda.assert_not_called()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(selection.call_args_list, [
            mock.call("so101", solutions=True, include_clean_table=True),
            mock.call("rebot", solutions=True, include_clean_table=True),
        ])
        self.assertIn("2 passed, 0 failed, 2 skipped", log.getvalue())
        self.assertIn("remain unverified", log.getvalue())
        self.assertNotIn("thousands", log.getvalue())

    def test_no_cuda_auto_skip_and_failures_remain_visible(self):
        checks = [final_check.Check("CPU", Path("cpu.py"), []),
                  final_check.Check("GPU", Path("gpu.py"), [], requires_cuda=True)]
        log = io.StringIO()
        with mock.patch.object(final_check, "checks_for", return_value=checks), \
             mock.patch.object(final_check, "cuda_available", return_value=False), \
             mock.patch.object(final_check, "run_check", return_value=(False, "unfinished TODO")), \
             contextlib.redirect_stdout(log):
            self.assertFalse(final_check.run_final_check())
        self.assertIn("0 passed, 1 failed, 1 skipped", log.getvalue())
        self.assertIn("unfinished TODO", log.getvalue())

    def test_an_entirely_skipped_suite_does_not_pass(self):
        checks = [final_check.Check("GPU", Path("gpu.py"), [], requires_cuda=True)]
        with mock.patch.object(final_check, "checks_for", return_value=checks), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(final_check.run_final_check(skip_gpu=True))

    def test_cpu_newton_gate_is_explicit_even_on_a_gpu_host(self):
        check = final_check.checks_for("so101")[-1]
        self.assertIn("--device", check.args)
        self.assertEqual(check.args[check.args.index("--device") + 1], "cpu")

    def test_clean_table_device_is_pinned_and_skip_gpu_takes_precedence(self):
        for skip_gpu, has_cuda, device, expected_device in (
            (True, True, "cuda:1", "cpu"),
            (False, False, None, "cpu"),
            (False, True, None, "cuda:0"),
            (False, True, "cuda:1", "cuda:1"),
        ):
            with self.subTest(skip_gpu=skip_gpu, has_cuda=has_cuda, device=device):
                with mock.patch.object(final_check, "cuda_available", return_value=has_cuda), \
                     mock.patch.object(final_check, "run_check", return_value=(True, "fixture passed")) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(final_check.run_final_check(
                        skip_gpu=skip_gpu, include_clean_table=True, include_newton_cuda=True, device=device,
                    ))
                checks = [call.args[0] for call in run.call_args_list]
                clean = checks[-1]
                self.assertEqual(clean.args.count("--device"), 1)
                self.assertEqual(clean.args[clean.args.index("--device") + 1], expected_device)
                self.assertEqual(clean.device, expected_device)
                artifact = Path(clean.args[clean.args.index("--report") + 1])
                self.assertEqual(artifact.stem.endswith("_cuda1"),
                                 device == "cuda:1" and expected_device == "cuda:1")
                self.assertEqual(sum(check.requires_cuda for check in checks),
                                 3 if has_cuda and not skip_gpu else 0)

    def test_cli_exposes_modes_and_returns_failure_exit_status(self):
        with mock.patch.object(sys, "argv", ["final_check.py", "--skip-gpu", "--solutions", "--robot", "both", "--include-clean-table"]), \
             mock.patch.object(final_check, "run_final_check", return_value=False) as run:
            with self.assertRaises(SystemExit) as context:
                final_check.main()
        self.assertEqual(context.exception.code, 1)
        run.assert_called_once_with(skip_gpu=True, robot="both", solutions=True, include_clean_table=True)

    def test_cli_selects_newton_cuda_and_device(self):
        with mock.patch.object(sys, "argv", ["final_check.py", "--solutions", "--robot", "both",
                                             "--include-newton-cuda", "--device", "cuda:1"]), \
             mock.patch.object(final_check, "run_final_check", return_value=True) as run:
            with self.assertRaises(SystemExit) as context:
                final_check.main()
        self.assertEqual(context.exception.code, 0)
        run.assert_called_once_with(skip_gpu=False, robot="both", solutions=True, include_clean_table=False,
                                    include_newton_cuda=True, device="cuda:1")

    def test_unknown_robot_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            final_check.checks_for("typo")
        with self.assertRaises(ValueError):
            final_check.run_final_check(skip_gpu=True, robot="typo")


if __name__ == "__main__":
    unittest.main()
