"""Real-window lifecycle of the Article 2 lesson scripts under an X server.

Runs the actual solution scripts (real MuJoCo/MJWarp physics, real GLFW window,
real OpenGL) and drives them like a user: XTEST key presses, a window-manager
close request and terminal-style SIGINT to the process group. Xvfb + Mesa is
enough; see ``tests/x11_lifecycle_harness.py``.

Skips cleanly when there is no DISPLAY / xdotool. Part 2 cases also need a
CUDA device (set ARTICLE2_X11_DEVICE, default cuda:0) and are skipped without
one. Robot assets are downloaded on first use (MUJOCO_MENAGERIE_CACHE).

    Xvfb :99 -screen 0 1600x1000x24 & export DISPLAY=:99
    notebooks/mujoco/.venv/bin/python -m unittest -v tests.test_article2_viewer_x11
"""
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x11_lifecycle_harness import X11App, x11_available  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SKIP = x11_available()
GL_ERRORS = ("Traceback", "X Error", "GLXBadDrawable", "Fatal glibc error", "Segmentation fault")


def cuda_device() -> str | None:
    device = os.environ.get("ARTICLE2_X11_DEVICE", "cuda:0")
    try:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        return device if wp.is_cuda_available() and wp.get_device(device).is_cuda else None
    except Exception:  # noqa: BLE001
        return None


class _Lifecycle:
    part: str
    script: str
    extra: tuple = ()
    episode_s = 12.0  # default --task stack episode

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="a2-x11-"))

    def launch(self, name):
        argv = [sys.executable, self.script, *self.extra]
        app = X11App(argv, cwd=ROOT / "notebooks/mujoco" / self.part, log_path=self.tmp / f"{name}.log",
                     window_name="MuJoCo teaching example")
        self.addCleanup(app.stop)
        return app.start()

    def window(self, app):
        wid = app.wait_window(240)
        self.assertIsNotNone(wid, app.output)
        time.sleep(1.0)
        return wid

    def assert_clean(self, app, code, expected):
        out = app.output
        self.assertEqual(code, expected, out)
        for marker in GL_ERRORS:
            self.assertNotIn(marker, out)
        self.assertTrue(app.windows_gone(), "window left behind")

    def test_completion_verifies_closes_and_exits_0(self):
        app = self.launch("complete")
        self.window(app)
        code = app.wait_exit(self.episode_s * 4 + 30)
        self.assert_clean(app, code, 0)
        self.assertIn("Episode finished", app.output)
        self.assertRegex(app.output, r"Task complete: red cube stacked on blue \(verified")

    def test_escape_q_and_window_close_quit(self):
        for action in ("Escape", "q", "close"):
            with self.subTest(action=action):
                app = self.launch(f"quit-{action}")
                wid = self.window(app)
                t0 = time.monotonic()
                if action == "close":
                    app.close_window(wid)
                else:
                    app.key(wid, action)
                code = app.wait_exit(10)
                self.assertLess(time.monotonic() - t0, 2.0, app.output)
                self.assert_clean(app, code, 0)
                label = {"Escape": "Esc", "q": "Q", "close": "window close"}[action]
                self.assertIn(f"Stopped by {label} after", app.output)

    def test_space_pauses_physics(self):
        app = self.launch("pause")
        wid = self.window(app)
        app.key(wid, "space")
        self.assertTrue(app.wait_output(r"Paused at frame", 5), app.output)
        time.sleep(1.5)
        app.key(wid, "space")
        self.assertTrue(app.wait_output(r"Resumed at frame", 5), app.output)
        paused = re.search(r"Paused at frame (\d+)/", app.output).group(1)
        resumed = re.search(r"Resumed at frame (\d+)/", app.output).group(1)
        self.assertEqual(paused, resumed)
        app.key(wid, "Escape")
        self.assert_clean(app, app.wait_exit(10), 0)

    def test_ctrl_c_in_viewer_loop_exits_130_fast(self):
        app = self.launch("sigint-loop")
        self.window(app)
        t0 = time.monotonic()
        app.ctrl_c()
        code = app.wait_exit(5)
        self.assertLess(time.monotonic() - t0, 1.0, app.output)
        self.assert_clean(app, code, 130)
        self.assertIn("Stopped by Ctrl+C", app.output)

    def test_ctrl_c_during_setup_exits_130_fast(self):
        app = self.launch("sigint-setup")
        self.assertTrue(app.wait_output(r"Loaded: ", 240), app.output)
        t0 = time.monotonic()
        app.ctrl_c()
        code = app.wait_exit(5)
        self.assertLess(time.monotonic() - t0, 1.6, app.output)
        self.assert_clean(app, code, 130)

    def test_double_ctrl_c_exits_immediately(self):
        app = self.launch("sigint-double")
        self.window(app)
        t0 = time.monotonic()
        app.ctrl_c()
        time.sleep(0.05)
        app.ctrl_c()
        code = app.wait_exit(5)
        self.assertLess(time.monotonic() - t0, 1.0, app.output)
        # 130 from os._exit, or -SIGINT if the second press lands after Python
        # has already restored the default handler; a shell reports 130 for both.
        self.assertIn(code, (130, -2), app.output)
        self.assertTrue(app.windows_gone(), "window left behind")


@unittest.skipIf(SKIP, SKIP or "")
class Part1CpuLifecycle(_Lifecycle, unittest.TestCase):
    part, script = "part1", "solutions/so101_pick_place_solution.py"


DEVICE = None if SKIP else cuda_device()


@unittest.skipIf(SKIP or DEVICE is None, SKIP or "no CUDA device for MJWarp")
class Part2MjwarpLifecycle(_Lifecycle, unittest.TestCase):
    part, script = "part2", "solutions/so101_mjwarp_solution.py"
    extra = ("--device", DEVICE or "cuda:0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
