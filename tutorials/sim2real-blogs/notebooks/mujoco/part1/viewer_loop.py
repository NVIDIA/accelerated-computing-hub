"""Finite, cancellable passive-viewer loop for the teaching examples.

The caller owns physics and task validation. This helper never reports success
for the physical task, and does not change a simulation frame's implementation.
"""
from __future__ import annotations

import math
import time


def run_passive_frames(model, data, simulate_frame, *, frames, fps, configure_viewer=None) -> bool:
    """Run exactly ``frames`` simulation frames, or return False on cancellation.

    Space pauses both the controller and physics; Escape and Q request exit.
    Keyboard callbacks only set flags. The calling thread owns context exit.
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
    quit_requested = False
    interrupted = False
    completed = 0
    frame_period = 1.0 / fps

    def on_key(keycode):
        nonlocal paused, quit_requested
        if keycode == 32:  # Space
            paused = not paused
        elif keycode in (256, ord("q"), ord("Q")):  # Escape / Q
            quit_requested = True

    with launch(model, data, key_callback=on_key) as viewer:
        try:
            if configure_viewer is not None:
                # The synchronous renderer owns camera/options on this thread.
                # Keep the existing configuration interface for both lessons.
                with viewer.lock():
                    configure_viewer(viewer)
            print("Space = pause/resume, Esc or Q = cancel. The viewer closes when the task finishes.")
            while completed < frames and not quit_requested and viewer.is_running():
                deadline = time.perf_counter() + frame_period
                if not paused:
                    simulate_frame()
                    completed += 1
                if quit_requested or not viewer.is_running():
                    break
                # Keep the window responsive even when no physics frame runs.
                viewer.sync()
                if quit_requested or completed == frames or not viewer.is_running():
                    break
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            # Catch inside the context so its one normal exit closes the viewer.
            interrupted = True

    complete = completed == frames and not quit_requested and not interrupted
    if not complete:
        print(f"Cancelled after {completed}/{frames} simulation frames; task completion was not checked.")
    return complete
