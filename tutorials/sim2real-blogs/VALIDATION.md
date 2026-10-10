# Validation scope

This update relocates the reviewed Article 2 source into the Hub tutorial. Its
physics scripts and dependency lock are unchanged; notebooks add Hub checkout
setup, use the Hub naming convention, and retain their reference/task defaults.
The [migration manifest](notebooks/mujoco/MIGRATION_MANIFEST.json) maps every
imported file and records each adaptation.

## Imported workstation evidence

- [Task-specific CPU/GPU study](notebooks/mujoco/benchmark-results/2026-10-09-validated-box-gpu0/README.md): both robots, nine environment counts, one complete warm-up and five repetitions. **35 of 36 configurations passed**. The reBot GPU 512 configuration failed its physical gate and supplies no accepted timing or ratio.
- [Notebook execution](notebooks/mujoco/notebook-validation/2026-10-10/README.md): four source notebooks passed actual kernels, including both box lessons and measurement-disabled benchmark preflight. Fundamentals produced finite image pixels; this is not image visual QA.
- [Viewer lifecycle](notebooks/mujoco/VIEWER_VALIDATION.md): finite completion and clean cancellation on the original Linux NVIDIA desktop, with the exact source relationship documented. The ordinary-Python macOS launch guard has regression coverage, not macOS GUI validation.
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

The [fresh Hub container validation](hub-validation/2026-10-10/README.md) built the updated image and passed **151 tests with one optional viewer-integration skip**, including both robots on CPU/GPU and the real-kernel benchmark preflight. Its initial collection failure and corrected test-only rerun are preserved. Compose lifecycle, live Brev/Colab, lower-end GPU and macOS GUI validation remain unverified.
