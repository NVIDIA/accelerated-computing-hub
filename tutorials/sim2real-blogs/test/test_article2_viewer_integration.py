"""Opt-in real CPU teaching tests, isolated from other articles' module aliases.

A fake window exercises lifecycle decisions while MuJoCo runs actual physics.
This does not validate a graphics driver. Run separately with the Article 2
Python environment and RUN_ARTICLE2_VIEWER_CPU_TESTS=1. No receipt is written to
the repository; failures retain the child stdout/stderr in the test assertion.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = r'''
"""Real MuJoCo physics with a fake window: this does not validate GLX teardown."""
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__REPOSITORY_ROOT__)
spec = importlib.util.spec_from_file_location(
    "article2_cpu_viewer_lesson", ROOT / "notebooks/mujoco/part1/solutions/so101_pick_place_solution.py")
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)
import viewer_loop
import viewer_window


class Clock:
    now = 0.0
    def perf_counter(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class Window:
    def __init__(self, model, data, key_callback, cancel=False):
        self.data = data
        self.key_callback = key_callback
        self.cam = SimpleNamespace(lookat=np.zeros(3))
        self.syncs = self.closes = 0
        self.cancel = cancel
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closes += 1
    def lock(self):
        return nullcontext()
    def is_running(self):
        # Always open: only the lesson's own completion/cancel logic may exit.
        return True
    def sync(self):
        self.syncs += 1
        if self.cancel and self.syncs == 10:
            self.key_callback(256)


class Integration(unittest.TestCase):
    def run_lesson(self, robot, *, headless=False, cancel=False, task="stack"):
        robot_spec = lesson.get_robot(robot)
        box_task = lesson.BoxTask(robot_spec, fps=50) if task == "box" else None
        if box_task:
            robot_spec = box_task.spec
            xml = box_task.resolve_scene(None, explicit=False)
        else:
            xml = lesson.resolve_pick_place_scene(None, explicit=False, spec=robot_spec)
        constructor = lesson.mujoco.MjData
        data_objects, windows = [], []
        def data_factory(model):
            data = constructor(model)
            data_objects.append(data)
            return data
        def launch(model, data, *, key_callback):
            window = Window(model, data, key_callback, cancel)
            windows.append(window)
            return window
        output = io.StringIO()
        with patch.object(lesson.mujoco, "MjData", side_effect=data_factory), \
             patch.object(viewer_window, "launch", side_effect=launch), \
             patch.object(viewer_loop, "time", Clock()), redirect_stdout(output):
            lesson.run_demo(xml, robot_spec,
                headless_steps=(2000 if box_task else 600) if headless else 0,
                sim_substeps=20 if box_task else 10, box_task=box_task)
        data = data_objects[0]
        return data, windows, output.getvalue()

    def test_complete_interactive_matches_headless_exactly(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                reference, no_windows, reference_output = self.run_lesson(robot, headless=True)
                actual, windows, output = self.run_lesson(robot)
                self.assertEqual(no_windows, [])
                self.assertEqual(len(windows), 1)
                self.assertEqual((windows[0].syncs, windows[0].closes), (600, 1))
                for field in ("qpos", "qvel", "ctrl", "xpos"):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))
                self.assertEqual(actual.time, reference.time)
                self.assertAlmostEqual(actual.time, 12.0, places=9)
                summary = [line for line in output.splitlines() if line.startswith("stack check:")]
                self.assertEqual(summary, [line for line in reference_output.splitlines() if line.startswith("stack check:")])
                self.assertEqual(len(summary), 1)

    def test_cancel_does_not_claim_completion(self):
        data, windows, output = self.run_lesson("so101", cancel=True)
        self.assertEqual((windows[0].syncs, windows[0].closes), (10, 1))
        self.assertAlmostEqual(data.time, 0.2)
        self.assertIn("Cancelled after 10/600", output)
        self.assertNotIn("stack check:", output)
        self.assertNotIn("Task complete", output)
        self.assertNotIn("PASS", output)

    def test_box_completion_preserves_actual_physical_checks(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                reference, _, _ = self.run_lesson(robot, headless=True, task="box")
                actual, windows, output = self.run_lesson(robot, task="box")
                self.assertEqual((windows[0].syncs, windows[0].closes), (2000, 1))
                for field in ("qpos", "qvel", "ctrl", "xpos"):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))
                self.assertEqual(actual.time, reference.time)
                self.assertAlmostEqual(actual.time, 40.0, places=8)
                self.assertIn("box check: PASS", output)
                self.assertNotIn("Cancelled", output)


if __name__ == "__main__":
    unittest.main()

'''


@unittest.skipUnless(os.environ.get("RUN_ARTICLE2_VIEWER_CPU_TESTS") == "1",
                     "Set RUN_ARTICLE2_VIEWER_CPU_TESTS=1 for native CPU integration")
class Article2ViewerIntegrationTests(unittest.TestCase):
    def test_native_cpu_completion_cancellation_and_box_checks(self):
        with tempfile.TemporaryDirectory(prefix="article2-viewer-test-") as directory:
            script = Path(directory) / "native_viewer_checks.py"
            script.write_text(SCRIPT.replace("__REPOSITORY_ROOT__", repr(str(ROOT))))
            environment = dict(os.environ, PXR_WORK_THREAD_LIMIT="1",
                               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                               PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-B", str(script)],
                                    env=environment, capture_output=True,
                                    text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Ran 3 tests", result.stderr)


if __name__ == "__main__":
    unittest.main()
