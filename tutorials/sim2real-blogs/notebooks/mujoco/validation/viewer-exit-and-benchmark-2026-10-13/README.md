# Viewer exit, Ctrl+C and benchmark output — validation, 13 October 2026

Response to a tester report: Part 1 stayed open after stacking, Space/Esc did nothing, Ctrl+C left a traceback followed by `GLXBadDrawable` and a hung process; the blog's Part 2 command looked like Part 1 and the benchmark numbers were unclear.

## Environment

- Linux workstation, 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (one used per run), AMD Ryzen Threadripper PRO 9985WX.
- Article 2 hash lock: MuJoCo 3.8.0, MuJoCo Warp 3.8.0.3, Warp 1.15.0, Python 3.12. The tester's copy was also run with its own pins (MuJoCo 3.12.0, MuJoCo Warp 3.12.0, Warp 1.17.0).
- Real X server: Xvfb 21.1 with Mesa llvmpipe OpenGL (no NVIDIA X display was available: no desktop session on the machine). Keys go through XTEST (`xdotool key`), window close is a `WM_DELETE_WINDOW` client message (what a window manager sends), Ctrl+C is SIGINT to the process group (what a terminal sends). Harness: `tests/x11_lifecycle_harness.py`.

## Before: the reported behaviour reproduces

`x11-scenarios-before-after.jsonl`, stack task, SO-101, one record per scenario (`rc: null` = still running when the harness killed it):

| Code | Finish | Esc | Q | Close window | Ctrl+C in loop | Ctrl+C after the stack |
| --- | --- | --- | --- | --- | --- | --- |
| Tester's copy (`ach-p1`, NVIDIA/accelerated-computing-hub `tutorials/sim2real-blogs`, 8 Oct) | never exits | ignored | ignored | exits | traceback | traceback, then `Fatal glibc error` abort |
| `cleanup/publish-ready-blog-tutorials` (`old-p1`) | never exits | ignored | ignored | exits | traceback + `Fatal glibc error` abort | traceback |
| `main` 258b863 (`main-p1`) | exits after ~70 s for a 12 s episode on software OpenGL | exits | exits | exits | exits, status 0 | exits, status 0 |
| This change (`fix-p1`) | exits 0 after ~15 s with `Task complete: … (verified …)` | exits 0 | exits 0 | exits 0 | exits 130 in 0.16 s | already exited by itself |

The old code used `mujoco.viewer.launch_passive`: an unbounded `while viewer.is_running()` loop, a render thread owned by MuJoCo, and a KeyboardInterrupt raised while that thread may still be swapping buffers. Under Mesa the race ends in a glibc assertion. The reported `GLXBadDrawable` followed by a process that ignores further Ctrl+C is, most likely, the same race under the NVIDIA GLX driver (not reproducible here without an NVIDIA X display).

`ctrl-c-during-warp-compile.json`: Part 2 from a cold Warp kernel cache, Ctrl+C 2 s and 6 s after start. `main`: traceback, 1.3 s and 8.9 s to exit. This change: status 130, no traceback, 1.1 s (grace watchdog) and 0.16 s.

## After: automated lifecycle tests

`lifecycle-tests.txt`: two consecutive runs, **47 passed** each:

- `tests/test_article2_viewer_x11.py` (real window, real physics), Part 1 CPU and Part 2 MJWarp on `cuda:0`: completion verifies, closes and exits 0; Esc, Q and window close exit 0 within 2 s; Space pauses (the frame counter at pause equals the one at resume); Ctrl+C in the viewer loop exits 130 in under 1 s; Ctrl+C during setup exits 130; a double Ctrl+C exits immediately; no `Traceback`, `X Error`, `GLXBadDrawable` or glibc error; no window left behind.
- `tests/test_article2_ctrl_c.py` (no display, CI-friendly): real SIGINTs to child processes for the setup phase, the polling loop, a teardown that never finishes (forced after the 1 s grace) and a second Ctrl+C; exit codes; catch-up pacing; pause; the Ctrl+C flag.
- Existing `test_article2_viewer_loop.py`, `test_article2_viewer_window.py` and the opt-in real-physics `test_article2_viewer_integration.py`.

`headless-matrix.txt`: both parts × both robots × stack and box, headless: every run exits 0 with a verified final line. Untouched student scaffolds still fail at their first TODO (exit 1).

Full `unittest discover` in the Article 2 environment: the only difference from `main` is one test fixed (`test_mjwarp_scaffold_refreshes_host_kinematics_before_check`, which already failed on `main`); the remaining failures need the separate Article 3 Newton environment or `pytest`, identical on both. With `pytest` installed, every Article 2 test module passes.

## Benchmark output

`scaling-study-rtx-pro-6000.txt`: `scaling_study.py` for both robots up to 32,768 worlds plus one `--benchmark --nworld 2048`, with the new self-describing header and reading guide. reBot at 16,384 worlds ran out of memory on the 96 GB GPU (about 4.6 MiB per world at the peak); the sweep now stops with an explanation instead of a traceback. The interpretation and the RTX 5090 / RTX 3060 expectations are in [`../../README.md#reading-the-benchmark-numbers`](../../README.md#reading-the-benchmark-numbers).

## Not covered

- macOS: not available; the macOS main-thread guard and "use `python`, not `mjpython`" instruction are unchanged.
- NVIDIA GLX under X11 with a desktop session: not available on this machine during the run; the earlier NVIDIA desktop validation in [`../viewer-lifecycle-2026-10-10/`](../viewer-lifecycle-2026-10-10/README.md) covered the same window code.
- Windows: not run.
