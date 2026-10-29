# Validation scope

The original import relocated the reviewed Article 2 source into the Hub tutorial.
The October 29 revision shares the CPU/GPU task implementation, moves source setup
into a helper, and transfers GPU controls and state once per control frame.
The dependency lock and task acceptance rules are unchanged. Archived measurements
refer to their recorded source versions; they are not measurements of this revision.
The [migration manifest](notebooks/mujoco/MIGRATION_MANIFEST.json) maps every
imported file and records each adaptation.

## Imported workstation evidence

- [Task-specific CPU/GPU study](notebooks/mujoco/benchmark-results/2026-10-09-validated-box-gpu0/README.md): both robots, nine environment counts, one complete warm-up and five repetitions. **35 of 36 configurations passed**. The reBot GPU 512 configuration failed its physical gate and supplies no accepted timing or ratio.
- [Notebook execution](notebooks/mujoco/notebook-validation/2026-10-10/README.md): four source notebooks passed actual kernels, including both box lessons and measurement-disabled benchmark preflight. Fundamentals produced finite image pixels; this is not image visual QA.
- [Viewer lifecycle](notebooks/mujoco/VIEWER_VALIDATION.md): finite completion and clean cancellation on the original Linux NVIDIA desktop, with the exact source relationship documented. The ordinary-Python macOS launch guard has regression coverage, not macOS GUI validation.
- [Viewer exit, Ctrl+C and benchmark output, 13 October](notebooks/mujoco/validation/viewer-exit-and-benchmark-2026-10-13/README.md): reproduces a tester's report against the previous `launch_passive` lessons under a real X server, and validates the fix (every run ends with a verified result and exit code 0/1, Ctrl+C exits 130 within about a second, self-describing benchmark output). Imported from [NVIDIA-dev/newton_blogs#2](https://github.com/NVIDIA-dev/newton_blogs/pull/2).
- [Legacy solver diagnostic](notebooks/mujoco/validation/legacy-linesearch-2026-10-10/README.md): original warning and diagnostic outcomes remain available. This is separate from the box benchmark.

All original archives, result JSON, figures and recorded hashes are unchanged.
They retain original source paths and historical links. Current tutorial links
and commands are in the [module README](notebooks/mujoco/README.md) and
[benchmark guide](notebooks/mujoco/BENCHMARK.md). These are community workstation
measurements; importing them is not a new hardware or container execution.

## Version boundary

The prior Hub [4 October validation record](history/VALIDATION-2026-10-04.md)
is preserved verbatim. It describes MuJoCo 3.12.0 / MuJoCo Warp 3.12.0 / Warp
1.17.0, including its own container checks. The current port intentionally uses
the source study's **3.8.0 / 3.8.0.3 / 1.15.0** environment and hash lock. The
historical container result cannot validate this updated image.

## Checks for this layout

With the tutorial's Python 3.12 locked environment active:

```bash
python -m pytest tutorials/sim2real-blogs/test -v
python brev/test-notebook-format.py sim2real-blogs
bash tutorials/sim2real-blogs/brev/test.bash
```

The full suite includes real task/recording checks and may download pinned
robot assets. CUDA checks skip explicitly if no GPU is available. Viewer unit
tests mock graphics; they do not open a desktop or establish GL validation.
Set `ARTICLE2_NOTEBOOKS_EXECUTE=1` to execute the benchmark notebook's preflight
in an actual kernel; measurements remain disabled. The optional CPU viewer
integration check uses `RUN_ARTICLE2_VIEWER_CPU_TESTS=1` and is not a GUI test.
`test/test_article2_viewer_x11.py` is a real-window GUI test: it runs the lesson scripts under an X server (Xvfb is enough) and sends real key presses, a window-manager close and Ctrl+C. It needs `DISPLAY` and `xdotool` and skips cleanly without them, so the container suite skips it. `test/test_article2_ctrl_c.py` checks the Ctrl+C exit contract with real signals and needs no display.

The [fresh Hub container validation](hub-validation/2026-10-10/README.md) built the updated image and passed **151 tests with one optional viewer-integration skip**, including both robots on CPU/GPU and the real-kernel benchmark preflight. Its initial collection failure and corrected test-only rerun are preserved. Compose lifecycle, live Brev/Colab, lower-end GPU and macOS GUI validation remain unverified.

## October 29 review corrections

Validated locally on macOS ARM64 with Python 3.12 and the Article 2 hash lock:

- Full tutorial suite: 169 passed, 17 skipped, 49 subtests passed. The final
  source-fingerprint change also passed all 41 migration-report tests.
- Executed notebooks 01, 02 and 04 in real local kernels. Both robots completed
  the full 40-second CPU box task; notebook 04 performed preflight only.
- GPU control-frame regression checks verify one control upload, 20 physics
  substeps, then one state/evidence download; capacity checks remain per substep.
- All four notebooks pass the Hub canonical-format check; `git diff --check` passes.

CUDA execution, GPU timings, the interactive viewer, Docker, Brev and Colab were
not rerun for this revision. Historical benchmark rows are unchanged. The hosted
bootstrap defaults to NVIDIA's `main` and will use these changes after merge;
reviewers can select the PR ref with `HUB_SOURCE_REF` before running setup.
