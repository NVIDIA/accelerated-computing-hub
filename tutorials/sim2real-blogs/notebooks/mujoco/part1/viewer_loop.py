"""Finite, cancellable teaching-viewer loop and Ctrl+C handling for the lessons.

The caller owns physics and task validation. This helper never reports success
for the physical task, and does not change a simulation frame's implementation.

Exit contract shared by every Article 2 lesson script (see ``lesson_main``):

* 0   episode finished and the task check passed, or the user quit the viewer
      (Esc, Q or closing the window; the task check is then skipped);
* 1   the task check failed, or the script raised an error;
* 130 Ctrl+C. A second Ctrl+C, or a first one that cannot stop the program
      within ``CTRL_C_GRACE_S``, terminates the process immediately.
"""
from __future__ import annotations

import math
import os
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager

EXIT_OK = 0
EXIT_TASK_FAILED = 1
EXIT_INTERRUPTED = 130
CTRL_C_GRACE_S = 1.0
CONTROLS_HINT = "Controls: Space pause/resume · Esc/Q quit · close the window to quit · Ctrl+C here to quit"

# Frames simulated back to back when rendering is slower than real time
# (software OpenGL, small laptop GPUs). Physics is never skipped; when even
# this is not enough the episode simply runs slower than real time.
MAX_CATCH_UP_FRAMES = 5
MAX_LAG_S = 0.25


class _CtrlC:
    """Process-wide Ctrl+C state. Only the main thread installs handlers."""

    installed = False
    requested = False
    as_flag = False  # True while a loop polls ``requested`` instead.
    _wakeup = None  # socket pair kept alive for signal.set_wakeup_fd


def _force_exit(message: bytes) -> None:
    try:
        os.write(2, message)
    finally:
        os._exit(EXIT_INTERRUPTED)


def _watch_ctrl_c(receiver: socket.socket, grace_s: float) -> None:
    """Runs in a daemon thread; woken by the C-level handler at once.

    Python runs signal handlers only in the main thread between bytecodes, so a
    long native call (OpenGL swap, Warp kernel compilation) would delay them.
    ``signal.set_wakeup_fd`` writes the signal number from C immediately, which
    lets this thread enforce the grace period and the second Ctrl+C.
    """
    presses = 0
    timer = None
    while True:
        try:
            data = receiver.recv(64)
        except OSError:
            return
        if not data:
            return
        presses += data.count(signal.SIGINT)
        if presses >= 2:
            _force_exit(b"\nSecond Ctrl+C: exiting immediately.\n")
        if presses == 1 and timer is None:
            timer = threading.Timer(grace_s, _force_exit,
                                    args=(b"\nCtrl+C: cleanup took too long, exiting now.\n",))
            timer.daemon = True
            timer.start()


def install_ctrl_c(grace_s: float = CTRL_C_GRACE_S) -> bool:
    """Install the lesson Ctrl+C policy. Returns False when it cannot (not main thread)."""
    if _CtrlC.installed:
        return True
    if threading.current_thread() is not threading.main_thread():
        return False

    def on_sigint(signum, frame):
        if _CtrlC.requested:
            # Normally the watcher thread exits first; this covers platforms
            # without a working wakeup fd.
            _force_exit(b"\nSecond Ctrl+C: exiting immediately.\n")
        _CtrlC.requested = True
        if _CtrlC.as_flag:
            # os.write, not print: a handler may interrupt a print in progress.
            os.write(2, b"\nCtrl+C received: stopping and closing the viewer...\n")
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)
    try:
        receiver, sender = socket.socketpair()
        sender.setblocking(False)
        signal.set_wakeup_fd(sender.fileno(), warn_on_full_buffer=False)
        threading.Thread(target=_watch_ctrl_c, args=(receiver, grace_s),
                         name="lesson-ctrl-c", daemon=True).start()
        _CtrlC._wakeup = (receiver, sender)  # keep the sockets alive
    except (OSError, ValueError):
        pass  # The Python handler above still provides the second-press exit.
    _CtrlC.installed = True
    return True


def ctrl_c_requested() -> bool:
    return _CtrlC.requested


@contextmanager
def ctrl_c_as_flag():
    """Inside this block Ctrl+C only sets a flag; the loop checks it every frame.

    That keeps KeyboardInterrupt out of OpenGL/GLFW calls, so the window is
    always torn down by normal code on the main thread (no swap after destroy).
    """
    previous = _CtrlC.as_flag
    _CtrlC.as_flag = True
    try:
        yield
    finally:
        _CtrlC.as_flag = previous


def lesson_main(main) -> None:
    """Run ``main()`` (which returns an exit code) under the lesson exit contract."""
    install_ctrl_c()
    try:
        code = main()
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C. Exiting.", flush=True)
        code = EXIT_INTERRUPTED
    if code is None:
        code = EXIT_OK
    sys.stdout.flush()
    sys.stderr.flush()
    if _CtrlC.requested and code != EXIT_INTERRUPTED:
        code = EXIT_INTERRUPTED
    if code == EXIT_INTERRUPTED:
        # The window is already closed and nothing is left to save: skip the
        # interpreter/CUDA/OpenGL finalizers so a Ctrl+C exit can never hang.
        os._exit(code)
    sys.exit(code)


class ViewerResult:
    """Outcome of an interactive episode. Truthy only for a complete episode."""

    __slots__ = ("completed_frames", "frames", "stopped_by")

    def __init__(self, completed_frames: int, frames: int, stopped_by: str | None):
        self.completed_frames = completed_frames
        self.frames = frames
        self.stopped_by = stopped_by  # None, "Esc", "Q", "window close" or "Ctrl+C"

    def __repr__(self) -> str:
        return f"ViewerResult({self.completed_frames}/{self.frames}, stopped_by={self.stopped_by!r})"

    @property
    def complete(self) -> bool:
        return self.stopped_by is None and self.completed_frames == self.frames

    @property
    def interrupted(self) -> bool:
        return self.stopped_by == "Ctrl+C"

    @property
    def exit_code(self) -> int:
        """Exit code when the task check is not run (cancelled episodes)."""
        return EXIT_INTERRUPTED if self.interrupted else EXIT_OK

    def __bool__(self) -> bool:
        return self.complete


def run_passive_frames(model, data, simulate_frame, *, frames, fps, configure_viewer=None) -> ViewerResult:
    """Run exactly ``frames`` simulation frames in a window, or stop early.

    Space pauses both the controller and physics; Escape, Q, closing the window
    and Ctrl+C stop. Keyboard callbacks and the Ctrl+C handler only set flags;
    the calling (main) thread checks them every frame and owns the teardown.
    The result is truthy only when every frame ran.
    """
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise ValueError("frames must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not callable(simulate_frame) or (configure_viewer is not None and not callable(configure_viewer)):
        raise TypeError("simulate_frame and configure_viewer must be callable")

    # Import only when an interactive run is requested. Headless runs do not
    # load this helper or acquire a window/context through it.
    from viewer_window import launch

    paused = False
    stopped_by = None
    completed = 0
    frame_period = 1.0 / fps

    def on_key(keycode):
        nonlocal paused, stopped_by
        if keycode == 32:  # Space
            paused = not paused
            print(f"Paused at frame {completed}/{frames}. Press Space to resume." if paused
                  else f"Resumed at frame {completed}/{frames}.", flush=True)
        elif keycode == 256:
            stopped_by = stopped_by or "Esc"
        elif keycode in (ord("q"), ord("Q")):
            stopped_by = stopped_by or "Q"

    def should_stop(viewer) -> bool:
        nonlocal stopped_by
        if stopped_by is None and ctrl_c_requested():
            stopped_by = "Ctrl+C"
        if stopped_by is None and not viewer.is_running():
            stopped_by = "window close"
        return stopped_by is not None

    with ctrl_c_as_flag(), launch(model, data, key_callback=on_key) as viewer:
        try:
            if configure_viewer is not None:
                # The synchronous renderer owns camera/options on this thread.
                with viewer.lock():
                    configure_viewer(viewer)
            print(f"{CONTROLS_HINT}. The viewer closes by itself after "
                  f"{frames} frames ({frames / fps:.0f} simulated seconds).", flush=True)
            next_due = time.perf_counter()
            while not should_stop(viewer):
                if paused:
                    next_due = time.perf_counter() + frame_period
                else:
                    if time.perf_counter() - next_due > MAX_LAG_S:
                        next_due = time.perf_counter()  # run slower than real time
                    for _ in range(MAX_CATCH_UP_FRAMES):
                        simulate_frame()
                        completed += 1
                        next_due += frame_period
                        if completed == frames or stopped_by or ctrl_c_requested():
                            break
                        if time.perf_counter() < next_due:
                            break
                    if should_stop(viewer):
                        break
                # Keep the window responsive even when no physics frame runs.
                viewer.sync()
                if should_stop(viewer) or completed == frames:
                    break
                remaining = min(next_due - time.perf_counter(), frame_period)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            # Only reachable without install_ctrl_c (e.g. imported elsewhere):
            # catch inside the context so its one normal exit closes the viewer.
            stopped_by = "Ctrl+C"

    result = ViewerResult(completed, frames, None if completed == frames and stopped_by is None else stopped_by)
    if result.complete:
        print(f"Episode finished: {frames}/{frames} frames ({frames / fps:.1f} simulated seconds). "
              "Viewer closed.", flush=True)
    else:
        print(f"Stopped by {result.stopped_by or 'user'} after {completed}/{frames} frames "
              f"({completed / fps:.1f} of {frames / fps:.1f} simulated seconds). Viewer closed; "
              "the task check was skipped because the episode did not finish.", flush=True)
    return result
