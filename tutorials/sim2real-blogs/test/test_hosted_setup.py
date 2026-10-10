"""Exercise actual teaching setup cells with all external commands mocked."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "notebooks/mujoco"
NOTEBOOKS = ("part1/01__mujoco_fundamentals.ipynb", "part1/02__pick_and_place.ipynb",
             "part2/03__mujoco_warp.ipynb")
SHA = "a" * 40
REPO_URL = "https://github.com/johnnynunez/accelerated-computing-hub.git"


class HostedSetupTests(unittest.TestCase):
    def execute(self, notebook, *, remote=REPO_URL, head=SHA, current_files=True,
                python_version=(3, 12), loaded_physics=False):
        cell = next(c for c in json.loads((ROOT / notebook).read_text())["cells"]
                    if c.get("id") == "hub-source-setup")
        tree = ast.parse("".join(cell["source"]))
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            lesson = checkout / "tutorials/sim2real-blogs/notebooks/mujoco" / Path(notebook).parent
            if current_files:
                lesson.mkdir(parents=True)
                (lesson / "viewer_window.py").write_text("# fixture marker\n")
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "COLAB_CHECKOUT" for t in node.targets):
                    node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(checkout))], keywords=[])
            ast.fix_missing_locations(tree)
            google, colab = ModuleType("google"), ModuleType("google.colab")
            google.colab = colab
            def output(command, **kwargs):
                calls.append(command)
                return remote + "\n" if command[-3:] == ["remote", "get-url", "origin"] else SHA + "\n"
            def run(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, stdout=head + "\n", stderr="")
            previous = Path.cwd()
            physics = {k: sys.modules.pop(k) for k in ("mujoco", "mujoco_warp", "warp") if k in sys.modules}
            try:
                if loaded_physics:
                    sys.modules["mujoco"] = ModuleType("mujoco")
                with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                     patch.object(sys, "version_info", python_version), \
                     patch.object(subprocess, "check_output", side_effect=output), \
                     patch.object(subprocess, "run", side_effect=run), \
                     contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(tree, notebook, "exec"), {})
            finally:
                os.chdir(previous)
                for k in ("mujoco", "mujoco_warp", "warp"):
                    sys.modules.pop(k, None)
                sys.modules.update(physics)
        return calls

    def test_existing_other_remote_refused(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook), self.assertRaisesRegex(RuntimeError, "another repository"):
                self.execute(notebook, remote="https://example.invalid/another.git")

    def test_existing_other_revision_preserved(self):
        for notebook in NOTEBOOKS:
            with self.subTest(notebook=notebook), self.assertRaisesRegex(RuntimeError, "existing files were preserved"):
                self.execute(notebook, head="b" * 40)

    def test_old_hub_revision_without_new_viewer_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "current box lessons"):
            self.execute(NOTEBOOKS[0], current_files=False)

    def test_wrong_kernel_or_already_loaded_physics_refused_before_install(self):
        with self.assertRaisesRegex(RuntimeError, "Python 3.12 kernel"):
            self.execute(NOTEBOOKS[0], python_version=(3, 11))
        with self.assertRaisesRegex(RuntimeError, "Restart the runtime"):
            self.execute(NOTEBOOKS[0], loaded_physics=True)

    def test_matching_checkout_uses_hash_lock_without_reset(self):
        for notebook in NOTEBOOKS:
            calls = self.execute(notebook)
            installs = [c for c in calls if "pip" in c]
            self.assertEqual(len(installs), 1)
            self.assertIn("--require-hashes", installs[0])
            self.assertTrue(installs[0][-1].endswith("requirements.lock.txt"))
            self.assertFalse(any("reset" in c or "checkout" in c for c in calls))
