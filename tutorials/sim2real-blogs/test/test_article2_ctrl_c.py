"""Ctrl+C policy and viewer-loop logic, without a display (CI friendly).

Real child processes receive real SIGINTs here: the lesson scripts' exit
contract (130 on Ctrl+C, forced exit on a second Ctrl+C or when cleanup hangs)
is a process-level property that mocks cannot prove.
"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PART1 = ROOT / "notebooks/mujoco/part1"
spec = importlib.util.spec_from_file_location("article2_ctrl_c_viewer_loop", PART1 / "viewer_loop.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def stub(body: str) -> str:
    return textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(PART1)!r})
        from viewer_loop import install_ctrl_c, lesson_main, ctrl_c_as_flag, ctrl_c_requested
        install_ctrl_c()
    """) + textwrap.dedent(body)


@unittest.skipUnless(sys.platform.startswith("linux") or sys.platform == "darwin", "POSIX signals")
class CtrlCProcessTests(unittest.TestCase):
    def spawn(self, body):
        proc = subprocess.Popen([sys.executable, "-u", "-c", stub(body)], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
        self.addCleanup(self.kill, proc)
        line = proc.stdout.readline().decode()
        self.assertEqual(line.strip(), "READY", line)
        return proc

    @staticmethod
    def kill(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        proc.stdout.close()

    def finish(self, proc, timeout):
        start = time.monotonic()
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.fail(f"process still running {timeout} s after Ctrl+C")
        return proc.returncode, out.decode(), time.monotonic() - start

    def test_setup_phase_ctrl_c_exits_130_without_traceback(self):
        proc = self.spawn("""
            def main():
                print("READY", flush=True)
                while True:
                    time.sleep(0.05)   # stands for asset download / model compile
            lesson_main(main)
        """)
        t0 = time.monotonic()
        os.killpg(proc.pid, signal.SIGINT)
        code, out, _ = self.finish(proc, 5)
        self.assertEqual(code, 130, out)
        self.assertLess(time.monotonic() - t0, 1.0, out)
        self.assertIn("Interrupted by Ctrl+C", out)
        self.assertNotIn("Traceback", out)

    def test_polling_loop_stops_cleanly_on_first_ctrl_c(self):
        proc = self.spawn("""
            def main():
                with ctrl_c_as_flag():
                    print("READY", flush=True)
                    while not ctrl_c_requested():
                        time.sleep(0.02)   # one viewer frame
                print("loop left normally", flush=True)
                return 0
            lesson_main(main)
        """)
        t0 = time.monotonic()
        os.killpg(proc.pid, signal.SIGINT)
        code, out, _ = self.finish(proc, 5)
        self.assertEqual(code, 130, out)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertIn("loop left normally", out)
        self.assertNotIn("Traceback", out)

    def test_cleanup_that_never_finishes_is_forced_after_grace(self):
        proc = self.spawn("""
            def main():
                with ctrl_c_as_flag():
                    print("READY", flush=True)
                    while True:          # ignores the flag: a stuck teardown
                        time.sleep(0.05)
            lesson_main(main)
        """)
        t0 = time.monotonic()
        os.killpg(proc.pid, signal.SIGINT)
        code, out, _ = self.finish(proc, 6)
        elapsed = time.monotonic() - t0
        self.assertEqual(code, 130, out)
        self.assertGreaterEqual(elapsed, helper.CTRL_C_GRACE_S - 0.2)
        self.assertLess(elapsed, helper.CTRL_C_GRACE_S + 1.0)
        self.assertIn("exiting now", out)

    def test_second_ctrl_c_exits_immediately(self):
        proc = self.spawn("""
            def main():
                with ctrl_c_as_flag():
                    print("READY", flush=True)
                    while True:
                        time.sleep(0.05)
            lesson_main(main)
        """)
        os.killpg(proc.pid, signal.SIGINT)
        time.sleep(0.2)
        t0 = time.monotonic()
        os.killpg(proc.pid, signal.SIGINT)
        code, out, _ = self.finish(proc, 5)
        self.assertEqual(code, 130, out)
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertIn("Second Ctrl+C", out)

    def test_exit_codes_follow_main(self):
        for value, expected in ((0, 0), (1, 1), (None, 0)):
            with self.subTest(value=value):
                proc = subprocess.run([sys.executable, "-c", stub(f"""
                    print("READY", flush=True)
                    lesson_main(lambda: {value!r})
                """)], capture_output=True, timeout=30)
                self.assertEqual(proc.returncode, expected, proc.stdout + proc.stderr)


class Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds


class Viewer:
    def __init__(self, render_cost=0.0, clock=None):
        self.render_cost, self.clock = render_cost, clock
        self.syncs, self.exits, self.callback, self.on_sync = 0, 0, None, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exits += 1

    def is_running(self):
        return True

    def lock(self):
        return contextlib.nullcontext()

    def sync(self):
        self.syncs += 1
        self.clock.now += self.render_cost
        if self.on_sync:
            self.on_sync()


class LoopLogicTests(unittest.TestCase):
    def run_loop(self, viewer, clock, frames=100, fps=50, step=None):
        module = ModuleType("viewer_window")
        def launch(model, data, *, key_callback):
            viewer.callback = key_callback
            return viewer
        module.launch = launch
        counter = {"frames": 0}
        def simulate():
            counter["frames"] += 1
            if step:
                step(counter["frames"])
        out = io.StringIO()
        with patch.dict(sys.modules, {"viewer_window": module}), patch.object(helper, "time", clock), \
                contextlib.redirect_stdout(out):
            result = helper.run_passive_frames(object(), object(), simulate, frames=frames, fps=fps)
        return result, counter["frames"], out.getvalue()

    def test_slow_rendering_catches_up_instead_of_slowing_physics(self):
        clock = Clock()
        viewer = Viewer(render_cost=0.1, clock=clock)  # 10 fps software OpenGL
        result, frames, out = self.run_loop(viewer, clock, frames=100)
        self.assertTrue(result.complete)
        self.assertEqual(frames, 100)
        # 2 s of simulated time renders in far fewer than 100 syncs, and the
        # wall clock stays close to real time (not 100 x 0.1 s = 10 s).
        self.assertLessEqual(viewer.syncs, 100 // helper.MAX_CATCH_UP_FRAMES + 2)
        self.assertLess(clock.now, 2.0 * 1.5)
        self.assertIn("Episode finished: 100/100 frames", out)

    def test_pause_freezes_frame_counter(self):
        clock = Clock()
        viewer = Viewer(clock=clock)
        def on_sync():
            if viewer.syncs == 10:
                viewer.callback(32)
            elif viewer.syncs == 40:
                viewer.callback(32)
            elif viewer.syncs == 50:
                viewer.callback(256)
        viewer.on_sync = on_sync
        result, frames, out = self.run_loop(viewer, clock)
        self.assertFalse(result)
        self.assertEqual(result.stopped_by, "Esc")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Paused at frame 10/100", out)
        self.assertIn("Resumed at frame 10/100", out)
        self.assertEqual(frames, 20)

    def test_ctrl_c_flag_stops_loop_with_exit_130(self):
        clock = Clock()
        viewer = Viewer(clock=clock)
        def step(n):
            if n == 5:
                helper._CtrlC.requested = True
        self.addCleanup(setattr, helper._CtrlC, "requested", False)
        result, frames, out = self.run_loop(viewer, clock, step=step)
        self.assertEqual((result.stopped_by, result.exit_code, frames), ("Ctrl+C", 130, 5))
        self.assertEqual(viewer.exits, 1)
        self.assertIn("Stopped by Ctrl+C after 5/100", out)

    def test_flag_mode_only_inside_loop(self):
        clock = Clock()
        viewer = Viewer(clock=clock)
        seen = []
        viewer.on_sync = lambda: seen.append(helper._CtrlC.as_flag)
        self.assertFalse(helper._CtrlC.as_flag)
        self.run_loop(viewer, clock, frames=3)
        self.assertEqual(seen, [True, True, True])
        self.assertFalse(helper._CtrlC.as_flag)


if __name__ == "__main__":
    unittest.main()
