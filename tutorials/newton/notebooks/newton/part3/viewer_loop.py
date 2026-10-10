"""Finite teaching-viewer lifecycle; physics and task checks stay with examples."""
from __future__ import annotations

import math
import time


def run_viewer_frames(viewer, example_factory, *, frames, render_fps=None,
                      post_step=None, finish=None) -> bool:
    """Advance a fixed physics-frame budget, respecting public viewer controls.

    Rendering continues while paused; only successful calls to ``step`` consume
    the budget. A cancelled run never calls ``finish``. This function owns the
    viewer, including cleanup when example construction or validation raises.
    """
    completed = 0
    try:
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise ValueError("frames must be a positive integer")
        if render_fps is not None and (isinstance(render_fps, bool)
                or not isinstance(render_fps, (int, float))
                or not math.isfinite(render_fps) or render_fps <= 0):
            raise ValueError("render_fps must be finite and positive, or None")
        try:
            example = example_factory()
            if hasattr(viewer, "hide_loading_splash"):
                viewer.hide_loading_splash()
            while completed < frames and viewer.is_running():
                started = time.perf_counter()
                # should_step consumes the viewer's optional single-step request;
                # call it once per rendered frame, including when paused.
                if viewer.should_step():
                    example.step()
                    completed += 1
                    if post_step is not None:
                        post_step(example)
                if not viewer.is_running():
                    break
                example.render()
                if completed == frames or not viewer.is_running():
                    break
                if render_fps is not None:
                    remaining = 1.0 / render_fps - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
            if completed == frames:
                if finish is not None:
                    finish(example)
                return True
        except KeyboardInterrupt:
            pass  # Expected cancellation; the finally block still closes once.
        print(f"Cancelled after {completed}/{frames} simulation frames; task completion was not checked.")
        return False
    finally:
        viewer.close()


def run_stack_gl(viewer, example_factory, args) -> bool:
    """Finite visible-GL path for the optional stack lesson only.

    Preserve its --test post-step/final checks and Newton's finite-state checks.
    Headless/export paths and other examples keep their original runner.
    """
    import newton.examples

    def post_step(example):
        if args.test and hasattr(example, "test_post_step"):
            example.test_post_step()

    def finish(example):
        if args.test:
            has_post = hasattr(example, "test_post_step")
            has_final = hasattr(example, "test_final")
            if has_final:
                example.test_final()
            elif not has_post:
                raise NotImplementedError("Example does not have a test_final or test_post_step method")
            for name in ("state_0", "state_1", "model", "control", "contacts"):
                if hasattr(example, name):
                    bad = newton.examples.find_nan_members(getattr(example, name))
                    if bad:
                        raise ValueError(f"NaN members found in {name}: {bad}")
        else:
            print(f"Completed {args.num_frames} simulation frames; use --test for the physical stack check.")

    requested_fps = getattr(args, "render_fps", None)
    return run_viewer_frames(viewer, example_factory, frames=args.num_frames,
        render_fps=50 if requested_fps is None else requested_fps,
        post_step=post_step, finish=finish)
