"""doctor.py: every problem it reports comes with an exact fix; no false alarms."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("blogs_doctor", ROOT / "doctor.py")
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


def run(argv, env=None):
    out = io.StringIO()
    environ = {k: v for k, v in os.environ.items() if k not in ("MUJOCO_MENAGERIE_PATH", "NEWTON_MENAGERIE_PATH")}
    environ.update(env or {})
    with mock.patch.object(sys, "argv", ["doctor.py", *argv]), \
         mock.patch.dict("os.environ", environ, clear=True), contextlib.redirect_stdout(out):
        code = doctor.main()
    return code, out.getvalue()


class DoctorTests(unittest.TestCase):
    def test_lock_pins_and_install_command_match_the_quickstart(self):
        pins = doctor.pinned_versions("2")
        self.assertEqual(pins["mujoco"], "3.8.0")
        self.assertEqual(pins["warp-lang"], "1.15.0")
        cmd = doctor.install_command("2")
        self.assertIn("uv venv --allow-existing --python 3.12 notebooks/mujoco/.venv", cmd)
        self.assertIn("--require-hashes -r notebooks/mujoco/requirements.lock.txt", cmd)
        readme = (ROOT / "README.md").read_text()
        self.assertIn("uv venv --allow-existing --python 3.12 notebooks/mujoco/.venv && uv pip install --python "
                      "notebooks/mujoco/.venv/bin/python --require-hashes -r notebooks/mujoco/requirements.lock.txt", readme)
        self.assertIn("python doctor.py", readme)

    def test_missing_packages_fail_with_the_install_command(self):
        with mock.patch.object(doctor, "installed", return_value=None), \
             mock.patch.object(doctor, "check_gpu"), mock.patch.object(doctor, "check_assets") as assets:
            code, out = run(["--article", "2"])
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] Article 2 packages missing", out)
        self.assertIn("fix: ", out)
        self.assertIn("requirements.lock.txt", out)
        assets.assert_not_called()

    def test_missing_menagerie_override_is_a_warning_with_unset_and_offline_is_a_fix(self):
        if doctor.installed("mujoco") is None:
            self.skipTest("needs the Article 2 or 3 environment")
        with tempfile.TemporaryDirectory() as home:
            env = {"HOME": home, "MUJOCO_MENAGERIE_PATH": f"{home}/mujoco_menagerie",
                   "MUJOCO_MENAGERIE_CACHE": f"{home}/cache"}
            with mock.patch.object(doctor, "check_gpu"), \
                 mock.patch.object(doctor.subprocess, "run",
                                   side_effect=OSError("network is unreachable")):
                code, out = run([], env)
        self.assertIn("does not exist; it is ignored", out)
        self.assertIn("fix: unset MUJOCO_MENAGERIE_PATH", out)
        self.assertIn("is not reachable from this machine", out)
        self.assertIn("python doctor.py --fetch", out)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
