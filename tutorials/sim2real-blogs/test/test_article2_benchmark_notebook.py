# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""The optional benchmark notebook must keep Run All diagnostic by default."""

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


PART2 = Path(__file__).resolve().parents[1] / "notebooks" / "mujoco" / "part2"
SOURCE = PART2 / "04__cpu_gpu_benchmark.ipynb"


class Article2BenchmarkNotebookTests(unittest.TestCase):
    def test_run_all_defaults_to_preflight_without_measurements(self):
        """Execute the real cells with external commands intercepted.

        This is a notebook safety/CLI contract, not a physics or timing result.
        An unexpected clone, install, or actual measurement fails the test.
        """
        import nbformat

        notebook = nbformat.read(SOURCE, as_version=4)
        nbformat.validate(notebook)
        spec = importlib.util.spec_from_file_location("benchmark_notebook_cli", PART2 / "migration_benchmark.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        calls = []

        def intercepted_run(command, **kwargs):
            command = [str(item) for item in command]
            if command[0] == "git" and command[-2:] == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, stdout="fixture-source-revision\n", stderr="")
            if len(command) == 3 and command[:2] == [sys.executable, "-c"]:
                self.assertTrue(kwargs.get("check"))
                return subprocess.CompletedProcess(command, 0)
            if command[:2] == [sys.executable, str(PART2 / "migration_benchmark.py")]:
                parsed = cli.create_parser().parse_args(command[2:])
                self.assertTrue(parsed.preflight, "Default Run All launched measured workloads")
                self.assertTrue(kwargs.get("check"), "Notebook must propagate subprocess failures")
                calls.append(parsed)
                return subprocess.CompletedProcess(command, 0)
            self.fail(f"Unexpected external command from default Run All: {command}")

        original_find_spec = importlib.util.find_spec

        def local_environment(name, *args, **kwargs):
            return None if name == "google.colab" else original_find_spec(name, *args, **kwargs)

        old_cwd = Path.cwd()
        try:
            os.chdir(PART2)
            with mock.patch.object(subprocess, "run", side_effect=intercepted_run), \
                    mock.patch.object(importlib.util, "find_spec", side_effect=local_environment), \
                    redirect_stdout(io.StringIO()):
                scope = {"__name__": "__main__"}
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        self.assertIsNone(cell.execution_count)
                        self.assertFalse(cell.outputs)
                        exec(compile(cell.source, f"{SOURCE.name}:cell-{index}", "exec"), scope)
        finally:
            os.chdir(old_cwd)
        self.assertEqual(len(calls), 1, "Run All must execute exactly one preflight")

    @unittest.skipUnless(os.environ.get("ARTICLE2_NOTEBOOKS_EXECUTE") == "1",
                         "Set ARTICLE2_NOTEBOOKS_EXECUTE=1 for real notebook preflight execution")
    def test_preflight_runs_in_the_selected_kernel_without_measurements(self):
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager

        original = SOURCE.read_bytes()
        notebook = nbformat.read(SOURCE, as_version=4)
        # Test-only audit cell inspects the namespace and actual output files.
        notebook.cells.append(nbformat.v4.new_code_cell(
            "import json\n"
            "assert RUN_BENCHMARK is False\n"
            "assert not (REPORT_DIR / 'results.json').exists()\n"
            "preflight = json.loads((REPORT_DIR / 'preflight/preflight.json').read_text())\n"
            "assert preflight['status'] == 'preflight' and preflight['cases'] == []\n"
        ))
        output = PART2 / ".generated/notebooks/02_Notebook_CPU_GPU_Benchmark_preflight_test.ipynb"
        output.parent.mkdir(parents=True, exist_ok=True)
        manager = KernelManager(kernel_name="python3", ip="127.0.0.1")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                                resources={"metadata": {"path": str(PART2)}})
        try:
            client.execute(cleanup_kc=True)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            nbformat.write(notebook, output)
            self.assertEqual(SOURCE.read_bytes(), original, "Execution changed the published source notebook")
        self.assertIsNone(client.kc, "Notebook execution leaked its kernel client")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNotNone(cell.execution_count)
                self.assertFalse(any(item.output_type == "error" for item in cell.outputs))


if __name__ == "__main__":
    unittest.main()
