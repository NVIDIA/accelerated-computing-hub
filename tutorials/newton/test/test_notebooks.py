"""Notebook contract tests; enable real execution with NEWTON_NOTEBOOKS_EXECUTE=1.

Execution uses the current test interpreter via an explicit kernel command, saves
only executed copies under part3/.generated/notebooks, and never rewrites source
notebooks or student scripts, including the advanced coupled task.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PART3 = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"
FOUNDATIONS = "01__newton_fundamentals.ipynb"


def load_notebook(name):
    return json.loads((PART3 / name).read_text())


def source_text(notebook, kind=None):
    return "\n\n".join("".join(cell["source"]) for cell in notebook["cells"]
                       if kind is None or cell["cell_type"] == kind)


class FoundationNotebookTests(unittest.TestCase):
    def test_foundations_teach_and_run_newton_before_robot_migration(self):
        notebook = load_notebook(FOUNDATIONS)
        markdown = source_text(notebook, "markdown")
        code = source_text(notebook, "code")
        for heading in ("Outline", "What you will learn", "Setup", "files", "Troubleshooting", "Recap", "Next"):
            self.assertIn(heading, markdown)
        for concept in ("ModelBuilder", "SolverBase", "SolverXPBD", "SolverMuJoCo", "SolverVBD",
                        "CollisionPipeline", "notify_model_changed", "Sensors", "replicate", "CPU", "CUDA"):
            self.assertIn(concept, markdown + code)
        self.assertIn("REFERENCE = True", code)
        self.assertIn("02__mujoco_to_newton.ipynb", markdown)
        self.assertIn("assert", code)
        self.assertIn(".step(", code)
        self.assertNotIn("from newton_scene import", code)
        self.assertNotIn("newton.Solver`", markdown)
        self.assertNotIn("immutable", markdown)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


MIGRATION = "02__mujoco_to_newton.ipynb"


class MigrationNotebookTests(unittest.TestCase):
    def test_migration_preserves_external_exercises_and_covers_both_robots(self):
        path = PART3 / MIGRATION
        self.assertTrue(path.exists(), "Missing dedicated migration notebook")
        notebook = load_notebook(MIGRATION)
        markdown, code = source_text(notebook, "markdown"), source_text(notebook, "code")
        for heading in ("Outline", "What you will learn", "Setup", "files", "Troubleshooting", "Recap", "Next"):
            self.assertIn(heading, markdown)
        for step in range(10):
            self.assertIn(f"Step {step}.", markdown)
        for term in ("so101", "rebot", "REFERENCE = True", "sys.executable", "check=True", "NotImplementedError"):
            self.assertIn(term, code)
        self.assertIn("joint_target_q_start", markdown + code)
        self.assertIn("0.08", markdown + code)
        self.assertNotIn("joint_target_pos", markdown + code)
        self.assertNotIn("except Exception", code)
        self.assertNotIn("returncode != 0", code)
        self.assertNotIn("!python", code)
        self.assertIn("not backup.exists()", code)
        self.assertIn("solutions/so101_newton_solution.py", code)
        self.assertIn("04__final_check.ipynb", markdown)


FINAL = "04__final_check.ipynb"
ADVANCED = "03__clean_the_table.ipynb"
NOTEBOOKS = (FOUNDATIONS, MIGRATION, ADVANCED, FINAL)


class AdvancedNotebookTests(unittest.TestCase):
    def test_advanced_lesson_matches_the_learning_path(self):
        notebook = load_notebook(ADVANCED)
        markdown, code = source_text(notebook, "markdown"), source_text(notebook, "code")
        for heading in ("Outline", "What you will learn", "Setup", "files", "Troubleshooting", "Recap", "Next"):
            self.assertIn(heading, markdown)
        for concept in ("SolverMuJoCo", "SolverVBD", "SolverCoupledProxy", "shirt", "cable", "ownership"):
            self.assertIn(concept, markdown + code)
        self.assertIn("REFERENCE = True", code)
        self.assertIn("04__final_check.ipynb", markdown)
        self.assertIn("tool_withdrawn", code)
        self.assertIn("tool_clearance_m", code)
        self.assertIn("record_sha256", markdown)

    def test_advanced_run_rejects_bad_grasp_history_or_recording(self):
        """Exercise notebook safety with synthetic protocol data, not physics."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is required to exercise recording validation")
        from clean_table_report_fixture import make_gripper_report

        spec = importlib.util.spec_from_file_location("final_check", PART3 / "final_check.py")
        final_check = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"final_check": final_check}):
            spec.loader.exec_module(final_check)
            cell = next("".join(c["source"]) for c in load_notebook(ADVANCED)["cells"]
                        if c["cell_type"] == "code"
                        and "validate_clean_table_report" in "".join(c["source"]))
            for reference in (True, False):
                for fault in (None, "grasp", "binding", "missing_object"):
                    with self.subTest(reference=reference, fault=fault), \
                         tempfile.TemporaryDirectory() as directory, \
                         contextlib.redirect_stdout(io.StringIO()):
                        root = Path(directory)

                        def write_synthetic_run(command, **kwargs):
                            script = ("solutions/clean_the_table_solution.py" if reference
                                      else "clean_the_table.py")
                            self.assertEqual(command[:2], [sys.executable, script])
                            self.assertEqual(kwargs["cwd"], root)
                            record = Path(command[command.index("--record") + 1])
                            report = Path(command[command.index("--report") + 1])
                            self.assertFalse(record.exists())
                            self.assertFalse(report.exists())
                            objects = ["red_cube", "blue_cube", "cable", "shirt"]
                            if fault == "missing_object":
                                objects[-1] = "cable"
                            np.savez(record, phase=np.asarray(["carry"] * 4),
                                     active_object=np.asarray(objects))
                            payload = make_gripper_report()
                            payload.update(newton="1.6.0", robot_asset_format="USD",
                                           record_sha256=hashlib.sha256(record.read_bytes()).hexdigest())
                            if fault == "grasp":
                                payload["objects"]["shirt"]["grasped"] = False
                            if fault == "binding":
                                payload["record_sha256"] = "0" * 64
                            report.write_text(json.dumps(payload))
                            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

                        scope = {"PART": root, "Path": Path, "ROBOT": "so101",
                                 "REFERENCE": reference, "subprocess": subprocess,
                                 "sys": sys, "json": json, "np": np}
                        with mock.patch.object(subprocess, "run", side_effect=write_synthetic_run):
                            if fault is None:
                                exec(compile(cell, "advanced-report-cell", "exec"), scope)
                            else:
                                with self.assertRaises(AssertionError):
                                    exec(compile(cell, "advanced-report-cell", "exec"), scope)

    def test_advanced_baseline_rejects_unrelated_exceptions(self):
        notebook = load_notebook(ADVANCED)
        cell = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and '"--num-frames", "1"' in "".join(c["source"]))
        expected = {
            "NotImplementedError: Complete TODO Step 0: add_bin",
            "NotImplementedError: Complete TODO Step 1: add_garment",
            "NotImplementedError: Complete TODO Step 2: add_cable",
            "NotImplementedError: Complete TODO Step 3: build_coupled_solver",
            "NotImplementedError: Complete TODO Step 4: coupled substep loop",
        }
        unrelated = (
            "RuntimeError: Complete TODO Step 0: add_bin", "ModuleNotFoundError: newton",
            "SyntaxError: Complete TODO Step 2: add_cable", "NotImplementedError: another exercise",
            "NotImplementedError: Complete TODO Step 99: unknown",
            "NotImplementedError: Complete TODO Step 2: add_cable (unexpected detail)",
            "NotImplementedError: Complete TODO Step 0: add_bin\nRuntimeError: later failure", "",
        )
        students = {PART3 / name: (PART3 / name).read_bytes()
                    for name in ("clean_table_scene.py", "clean_the_table.py")}
        for error in (*sorted(expected), *unrelated):
            with self.subTest(error=error):
                failure = subprocess.CalledProcessError(1, [sys.executable], output="", stderr=error)
                with mock.patch.object(subprocess, "run", side_effect=failure) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    scope = {"PART": PART3, "subprocess": subprocess, "sys": sys, "REFERENCE": True}
                    if error in expected:
                        exec(compile(cell, "advanced-baseline-cell", "exec"), scope)
                    else:
                        with self.assertRaises(subprocess.CalledProcessError):
                            exec(compile(cell, "advanced-baseline-cell", "exec"), scope)
                    self.assertIs(run.call_args.kwargs.get("check"), True)
        for path, before in students.items():
            self.assertEqual(path.read_bytes(), before)


class FinalNotebookTests(unittest.TestCase):
    def test_final_notebook_asserts_both_robots_including_the_advanced_task(self):
        self.assertTrue((PART3 / FINAL).exists(), "Missing final notebook 04")
        self.assertFalse((PART3 / "02_Notebook_Final_Check.ipynb").exists(), "Obsolete notebook was not removed")
        notebook = load_notebook(FINAL)
        markdown, code = source_text(notebook, "markdown"), source_text(notebook, "code")
        for heading in ("Outline", "What you will learn", "Setup", "files", "Troubleshooting", "Recap", "Next"):
            self.assertIn(heading, markdown)
        self.assertIn("REFERENCE = True", code)
        module = ast.parse(code)
        calls = [node.test for node in ast.walk(module) if isinstance(node, ast.Assert)
                 and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                 and node.test.func.id == "run_final_check"]
        self.assertTrue(calls, "The final run must be asserted, not assigned and ignored")
        expected = {"skip_gpu": "True", "robot": "'both'", "solutions": "REFERENCE", "include_clean_table": "True"}
        self.assertIn(expected, [{kw.arg: ast.unparse(kw.value) for kw in call.keywords} for call in calls])
        for field in ("CLEAN_TABLE_RESULT", "inside", "settled_frames", "phase", "success"):
            self.assertIn(field, markdown)
        self.assertIn("remain unverified", markdown)

    def test_notebooks_are_valid_clean_sources_with_unique_cell_ids(self):
        ids = []
        for name in NOTEBOOKS:
            self.assertTrue((PART3 / name).is_file(), name)
            notebook = load_notebook(name)
            self.assertEqual((notebook["nbformat"], notebook["nbformat_minor"]), (4, 5))
            for cell in notebook["cells"]:
                self.assertTrue(cell.get("id"), (name, "missing id"))
                ids.append(cell["id"])
                if cell["cell_type"] == "code":
                    self.assertEqual(cell.get("outputs"), [], name)
                    self.assertIsNone(cell.get("execution_count"), name)
                    source = "".join(cell["source"])
                    self.assertNotIn("!python", source)
                    self.assertNotIn("%run", source)
                    ast.parse(source)
        self.assertEqual(len(ids), len(set(ids)), "Cell IDs must be unique across the lessons")


class NotebookSafetyTests(unittest.TestCase):
    def test_backup_cell_never_overwrites_existing_backups_or_student_work(self):
        notebook = load_notebook(MIGRATION)
        cell = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "not backup.exists()" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for name in ("newton_scene.py", "so101_newton.py"):
                (root / name).write_text("original student work")
            scope = {"PART3": root, "shutil": shutil}
            exec(compile(cell, "migration-backup-cell", "exec"), scope)
            for name in ("newton_scene.py", "so101_newton.py"):
                (root / name).write_text("new student work")
            exec(compile(cell, "migration-backup-cell", "exec"), scope)
            for name in ("newton_scene.py", "so101_newton.py"):
                student = root / name
                self.assertEqual(student.read_text(), "new student work")
                self.assertEqual(student.with_name(student.stem + "_original.py").read_text(), "original student work")

    def test_baseline_cell_accepts_only_the_explicit_incomplete_error(self):
        notebook = load_notebook(MIGRATION)
        cell = next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "sentinel =" in "".join(c["source"]))
        sentinel = 'raise NotImplementedError("Complete TODO Steps 1-5 in build_newton_model()")'
        expected = "NotImplementedError: Complete TODO Steps 1-5 in build_newton_model()"
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "newton_scene.py").write_text(sentinel)
            scope = {"PART3": root, "subprocess": subprocess, "sys": sys}
            for error in (expected, "ModuleNotFoundError: No module named 'newton'", "SyntaxError: broken"):
                failure = subprocess.CalledProcessError(1, [sys.executable], output="", stderr=error)
                with mock.patch.object(subprocess, "run", side_effect=failure) as run:
                    if error == expected:
                        exec(compile(cell, "migration-baseline-cell", "exec"), scope)
                    else:
                        with self.assertRaises(subprocess.CalledProcessError):
                            exec(compile(cell, "migration-baseline-cell", "exec"), scope)
                    self.assertIs(run.call_args.kwargs["check"], True)
                    self.assertEqual(run.call_args.args[0][0], sys.executable)
            with mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                with self.assertRaises(AssertionError):
                    exec(compile(cell, "migration-baseline-cell", "exec"), scope)

    def test_nbformat_schema(self):
        try:
            import nbformat
        except ImportError:
            self.skipTest("nbformat is not installed; use the dedicated Article 3 environment")
        for name in NOTEBOOKS:
            nbformat.validate(nbformat.read(PART3 / name, as_version=4))


@unittest.skipUnless(os.environ.get("NEWTON_NOTEBOOKS_EXECUTE") == "1",
                     "Set NEWTON_NOTEBOOKS_EXECUTE=1 in the pinned Article 3 environment for real execution")
class NotebookExecutionTests(unittest.TestCase):
    """No fake output, no source writes, no default-system-kernel substitution."""
    def execute_notebook(self, name):
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager

        source = PART3 / name
        source_bytes = source.read_bytes()
        student_files = [PART3 / name for name in (
            "newton_scene.py", "so101_newton.py", "clean_table_scene.py", "clean_the_table.py",
        )]
        students = {path: path.read_bytes() for path in student_files}
        notebook = nbformat.read(source, as_version=4)
        # Use loopback TCP, the tested local kernel transport; do not change
        # a notebook's interpreter by falling back to a system kernelspec.
        km = KernelManager(kernel_name="python3", ip="127.0.0.1")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        # Notebook 04 runs two separately bounded 1800 s gripper tasks in one
        # cell, plus rigid checks and compilation. Its cell must cover both.
        cell_timeout = 4200 if name == FINAL else 1900
        client = NotebookClient(notebook, km=km, timeout=cell_timeout, allow_errors=False,
                                resources={"metadata": {"path": str(PART3)}})
        output = PART3 / ".generated" / "notebooks" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Passing our own manager disables nbclient's default cleanup.
            # Opt in explicitly so its TCP channels do not leak across lessons.
            client.execute(cleanup_kc=True)
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
            nbformat.write(notebook, output)
            self.assertEqual(source.read_bytes(), source_bytes, "Execution rewrote lesson source")
            for path, original in students.items():
                self.assertEqual(path.read_bytes(), original, f"Execution overwrote {path}")
        self.assertIsNone(client.kc, "Notebook execution leaked its kernel client channels")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNotNone(cell.execution_count)
                self.assertFalse(any(item.output_type == "error" for item in cell.outputs))
        print(f"Executed {name}; saved actual outputs to {output}")

    def test_foundations_run_headless(self):
        self.execute_notebook(FOUNDATIONS)

    def test_both_robot_migrations_run_headless(self):
        self.execute_notebook(MIGRATION)

    @unittest.skipUnless(os.environ.get("NEWTON_NOTEBOOKS_INCLUDE_FINAL") == "1",
                         "Full coupled-task notebooks are opt-in: NEWTON_NOTEBOOKS_INCLUDE_FINAL=1")
    def test_clean_table_runs_headless(self):
        self.execute_notebook(ADVANCED)

    @unittest.skipUnless(os.environ.get("NEWTON_NOTEBOOKS_INCLUDE_FINAL") == "1",
                         "Final coupled-task notebook is opt-in: NEWTON_NOTEBOOKS_INCLUDE_FINAL=1")
    def test_final_includes_both_full_clean_table_tasks(self):
        self.execute_notebook(FINAL)


if __name__ == "__main__":
    unittest.main()
