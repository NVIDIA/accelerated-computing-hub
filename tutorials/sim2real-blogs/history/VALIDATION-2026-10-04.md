# Validation — 4 October 2026

These checks use this tutorial's Python 3.12, MuJoCo **3.12.0**, MuJoCo Warp
**3.12.0**, Warp **1.17.0** and NumPy **2.5.3** environment. The earlier companion
repository's MuJoCo 3.8 results were checked separately and are not substituted
for this version line.

## Task matrix

| Backend | Robot | Original stack | Receiving box |
|---|---|---|---|
| Native MuJoCo CPU | SO-101 | Pass, 600 frames | Pass, 1,800 frames |
| Native MuJoCo CPU | reBot | Pass, 600 frames | Pass, 1,800 frames |
| MJWarp CUDA 0 | SO-101 | Pass, 600 frames | Pass, 2,000 frames |
| MJWarp CUDA 0 | reBot | Pass, 600 frames | Pass, 2,000 frames |
| MJWarp CUDA 1 | SO-101 | Pass, 600 frames | Pass, 2,000 frames |
| MJWarp CUDA 1 | reBot | Pass, 600 frames | Pass, 2,000 frames |

CPU task checks ran on macOS ARM64. CUDA checks ran on Linux x86-64 with two
NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs and driver 615.71.09.
Each process selected one device; this is not a multi-GPU scaling benchmark.
The checked runtime snapshot contained 31 Python sources, all matching the
submitted versions.

The box runs were independently audited from measured 50 Hz observations.
Both cubes established opposing loaded jaw contacts, sustained whole-object
lift and airborne carry, opening over the box, detached release and settling,
whole-cube containment and final jaw withdrawal. GPU observations preserve
forces from the GPU solver. The contact adapter reads `contact.geom`, because
MJWarp exports that modern pair while legacy `geom1`/`geom2` can retain stale IDs.

Capacity checks run after each GPU substep. Peak box-task counts were 33 contacts
and 180 constraints for SO-101 (capacities 128/300), and 20/69 for reBot
(capacities 256/500). No capacity overflow occurred. Solver iteration-limit
flags are retained as convergence diagnostics and are not treated as capacity
flags. Passing the task does not establish full solver convergence or identical
CPU/GPU trajectories.

The published commands allow 2,000 frames: 40 simulated seconds at 50 Hz with
a 0.002 s physics timestep. That budget includes settling margin. During the
separate MuJoCo 3.8 validation, one reBot run at 36 seconds had only 35 of the
required 50 consecutive settled frames; the controller and acceptance thresholds
were left unchanged. All four final CUDA cases in the 3.12 matrix were checked
at 40 seconds, with at least 62 consecutive detached settled frames at the end.

## Regression and packaging checks

- Local pinned-environment suite: **24 passed, 3 CUDA checks explicitly skipped**.
- Dedicated Docker image built on Linux using the published CUDA 13.1/Python 3.12
  recipe. Its dependency check passed; the installed interpreter was Python 3.12.15.
- Inside that image, `bash tutorials/sim2real-blogs/brev/test.bash` passed
  **27 tests and 5 subtests, with no skips**, in 101.87 seconds. This includes the
  CUDA stack, captured benchmark, actual constraint-overflow rejection, both CPU
  box references and the new acceptance/contact-export regression tests.
- Both SO-101 and reBot box references also passed at 2,000 frames inside the
  image on one selected GPU, with JSON reports confirming task success.
- All three notebooks passed the Hub canonical-format checker. The two existing
  task notebooks retain all 18 previous code cells and each add one opt-in box cell.
- Python syntax, 26 relative documentation links, Compose configuration, shell
  syntax and `git diff --check` passed. The existing Warp tutorial tree matches
  the PR base.

Robot assets are downloaded from the pinned upstream revisions. The container
was tested directly with one GPU; a hosted Brev deployment, Compose service
lifecycle, interactive viewers and Colab execution were not tested. Colab links
point to upstream `main` and become usable after merge. Repository CI on this
fork still requires the maintainers' normal review process.
