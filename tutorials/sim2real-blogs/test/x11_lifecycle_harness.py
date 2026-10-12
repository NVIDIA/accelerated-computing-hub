"""Drive a teaching script under a real X server with real key and signal events.

Used by ``test_article2_viewer_x11.py`` and runnable by hand. Requirements:
``DISPLAY`` pointing at an X server (Xvfb is enough) and ``xdotool`` on PATH.
Key presses go through XTEST (``xdotool key``), i.e. the same path as a
physical keyboard. Window close sends ``WM_DELETE_WINDOW`` exactly like a
window manager's close button. Ctrl+C is ``SIGINT`` to the whole process
group, which is what a terminal does.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path


def x11_available() -> str | None:
    """Return None when usable, else a human-readable skip reason."""
    if not os.environ.get("DISPLAY"):
        return "DISPLAY is not set (start Xvfb, e.g. Xvfb :99 & export DISPLAY=:99)"
    if shutil.which("xdotool") is None:
        return "xdotool is not on PATH"
    if ctypes.util.find_library("X11") is None:
        return "libX11 is not available"
    probe = subprocess.run(["xdotool", "getdisplaygeometry"], capture_output=True, text=True)
    if probe.returncode != 0:
        return f"cannot reach X server {os.environ['DISPLAY']}: {probe.stderr.strip()}"
    return None


class _XClientMessage(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p), ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int), ("data", ctypes.c_long * 5),
    ]


class _XEvent(ctypes.Union):
    _fields_ = [("xclient", _XClientMessage), ("pad", ctypes.c_long * 24)]


def send_wm_delete(window_id: int) -> None:
    """Ask a window to close the way a window manager's close button does."""
    x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long,
                               ctypes.POINTER(_XEvent)]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("XOpenDisplay failed")
    try:
        event = _XEvent()
        event.xclient.type = 33  # ClientMessage
        event.xclient.window = window_id
        event.xclient.message_type = x11.XInternAtom(display, b"WM_PROTOCOLS", 0)
        event.xclient.format = 32
        event.xclient.data[0] = x11.XInternAtom(display, b"WM_DELETE_WINDOW", 0)
        event.xclient.data[1] = 0  # CurrentTime
        x11.XSendEvent(display, window_id, 0, 0, ctypes.byref(event))
        x11.XFlush(display)
    finally:
        x11.XCloseDisplay(display)


class X11App:
    """One child process in its own session/process group, output to a log file."""

    def __init__(self, argv, *, cwd, log_path: Path, env=None, window_name="MuJoCo"):
        self.argv, self.cwd, self.log_path = list(argv), cwd, Path(log_path)
        self.env = dict(os.environ if env is None else env)
        self.env.setdefault("PYTHONUNBUFFERED", "1")
        self.window_name = window_name
        self.proc: subprocess.Popen | None = None
        self.t0 = 0.0

    def start(self) -> "X11App":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        self.t0 = time.monotonic()
        self.proc = subprocess.Popen(self.argv, cwd=self.cwd, env=self.env, stdout=self._log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        return self

    @property
    def output(self) -> str:
        return self.log_path.read_bytes().decode("utf-8", "replace")

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def wait_output(self, pattern: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if re.search(pattern, self.output):
                return True
            if self.proc.poll() is not None:
                return bool(re.search(pattern, self.output))
            time.sleep(0.05)
        return False

    def window_ids(self) -> list[int]:
        res = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", self.window_name],
                             capture_output=True, text=True)
        return [int(w) for w in res.stdout.split()]

    def wait_window(self, timeout: float) -> int | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ids = self.window_ids()
            if ids:
                return ids[-1]
            if self.proc.poll() is not None:
                return None
            time.sleep(0.1)
        return None

    def windows_gone(self, timeout: float = 2.0) -> bool:
        """True once no window of ours is mapped. The X server destroys a dead
        client's windows asynchronously, so poll briefly after exit."""
        deadline = time.monotonic() + timeout
        while self.window_ids():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def key(self, window_id: int, keysym: str) -> None:
        subprocess.run(["xdotool", "windowfocus", "--sync", str(window_id)], check=False,
                       capture_output=True, timeout=5)
        subprocess.run(["xdotool", "key", "--clearmodifiers", keysym], check=True, timeout=5)

    def close_window(self, window_id: int) -> None:
        send_wm_delete(window_id)

    def ctrl_c(self) -> None:
        os.killpg(self.proc.pid, signal.SIGINT)

    def wait_exit(self, timeout: float) -> int | None:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self) -> None:
        """Hard cleanup for the test itself; never part of an assertion."""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=10)
        if getattr(self, "_log", None):
            self._log.close()
