"""Checkout preservation and locked-kernel setup, without external commands."""
import contextlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "notebooks/mujoco"
SPEC = importlib.util.spec_from_file_location("article2_setup", ROOT / "notebook_setup.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)
SHA = "a" * 40


class HostedSetupTests(unittest.TestCase):
    def checkout(self, *, remote=setup.REPO_URL, head=SHA):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=head + "\n", stderr="")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(subprocess, "check_output", return_value=remote + "\n"), \
                patch.object(subprocess, "run", side_effect=run):
            setup.source_checkout(Path(directory), SHA)
        return calls

    def test_existing_other_remote_refused(self):
        with self.assertRaisesRegex(RuntimeError, "another repository"):
            self.checkout(remote="https://example.invalid/another.git")

    def test_existing_other_revision_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "existing files were preserved"):
            self.checkout(head="b" * 40)

    def test_matching_checkout_never_resets_or_checks_out(self):
        calls = self.checkout()
        self.assertFalse(any("reset" in c or "checkout" in c or "fetch" in c for c in calls))

    def test_unresolved_ref_refused(self):
        with self.assertRaises(ValueError):
            setup.source_checkout(Path("/not-created"), "main")

    def prepare(self, *, files=True, version=(3, 12), loaded=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lesson = root / setup.TUTORIAL_RELATIVE / "part1"
            lesson.mkdir(parents=True)
            if files:
                (lesson / "viewer_window.py").touch()
            previous = Path.cwd()
            physics = {k: sys.modules.pop(k) for k in ("mujoco", "mujoco_warp", "warp") if k in sys.modules}
            try:
                if loaded:
                    sys.modules["mujoco"] = object()
                with patch.dict(os.environ), patch.object(sys, "version_info", version), patch.object(subprocess, "run") as run:
                    setup.prepare_lesson(root, "part1", hosted=True)
                    return run.call_args.args[0]
            finally:
                os.chdir(previous)
                if str(lesson) in sys.path:
                    sys.path.remove(str(lesson))
                for k in ("mujoco", "mujoco_warp", "warp"):
                    sys.modules.pop(k, None)
                sys.modules.update(physics)

    def test_old_source_refused(self):
        with self.assertRaisesRegex(RuntimeError, "current box lessons"):
            self.prepare(files=False)

    def test_wrong_kernel_and_loaded_physics_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Python 3.12 kernel"):
            self.prepare(version=(3, 11))
        with self.assertRaisesRegex(RuntimeError, "Restart the runtime"):
            self.prepare(loaded=True)

    def test_installs_hash_lock(self):
        command = self.prepare()
        self.assertIn("--require-hashes", command)
        self.assertTrue(command[-1].endswith("requirements.lock.txt"))
