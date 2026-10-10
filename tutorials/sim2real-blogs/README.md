# Sim2Real Blogs

Companion tutorials for the simulation-to-real robotics blog series. Article 2
runs the same receiving-box task on native MuJoCo CPU and MuJoCo Warp GPU:
each arm grasps both cubes, carries and releases them into the box, then withdraws.
The lessons check both SO-101 and Seeed reBot DevArm. They do not deploy a policy
to a physical robot.

| Notebook | Description | Colab |
|---|---|---|
| [MuJoCo fundamentals](notebooks/mujoco/part1/01__mujoco_fundamentals.ipynb) | Inspect a model, state and CPU stepping. | [Open](https://colab.research.google.com/github/johnnynunez/accelerated-computing-hub/blob/feature/blog2-validated-box-benchmark/tutorials/sim2real-blogs/notebooks/mujoco/part1/01__mujoco_fundamentals.ipynb) |
| [Pick and place](notebooks/mujoco/part1/02__pick_and_place.ipynb) | Both robots place both cubes in the box. | [Open](https://colab.research.google.com/github/johnnynunez/accelerated-computing-hub/blob/feature/blog2-validated-box-benchmark/tutorials/sim2real-blogs/notebooks/mujoco/part1/02__pick_and_place.ipynb) |
| [MuJoCo Warp](notebooks/mujoco/part2/03__mujoco_warp.ipynb) | The same task on one selected GPU. | [Open](https://colab.research.google.com/github/johnnynunez/accelerated-computing-hub/blob/feature/blog2-validated-box-benchmark/tutorials/sim2real-blogs/notebooks/mujoco/part2/03__mujoco_warp.ipynb) |
| [CPU/GPU benchmark](notebooks/mujoco/part2/04__cpu_gpu_benchmark.ipynb) | Preflight first; optionally measure 1 through 2048 environments. | [Open](https://colab.research.google.com/github/johnnynunez/accelerated-computing-hub/blob/feature/blog2-validated-box-benchmark/tutorials/sim2real-blogs/notebooks/mujoco/part2/04__cpu_gpu_benchmark.ipynb) |

See the [lesson guide](notebooks/mujoco/README.md), [benchmark method and results](notebooks/mujoco/BENCHMARK.md),
and [validation scope](VALIDATION.md). Reference implementations run by default;
student exercises remain editable and optional stacking stays separate.

## Local and hosted setup

Use Python 3.12 with the module's unchanged hash lock. This update deliberately
uses the measured MuJoCo 3.8.0 / MuJoCo Warp 3.8.0.3 / Warp 1.15.0 line, replacing
the previous Hub tutorial's 3.12 / 1.17 dependencies. Newton uses a separate environment.

The existing Brev Docker Compose services and shared Hub entrypoint are retained.
The image installs the same physics lock before adding notebook/UI tools under
constraints. Build and test it through the Hub scripts:

```bash
brev/dev-build.bash sim2real-blogs
brev/dev-test.bash sim2real-blogs
```

A container build, Brev deployment or Colab run is separate validation; imported
workstation measurements do not establish those results. The Colab links select
the public contribution source before merge. Setup prints its resolved Git SHA,
refuses to overwrite a checkout at another revision, and skips LFS downloads for
teaching code. Teaching cells need a Python 3.12 kernel; the benchmark notebook
uses an isolated Python 3.12 environment for hosted subprocesses.

## Evidence and attribution

[Migration provenance](notebooks/mujoco/MIGRATION_MANIFEST.json) records relocation
and source identities. The original result, notebook, viewer and reference packages
are retained byte-for-byte, including failed cases and their historical paths.
Pull the required LFS objects when inspecting archived evidence; notebook physics
does not require downloading those archives.

The imported materials retain their [MIT license](notebooks/mujoco/LICENSE) and
[third-party notices](notebooks/mujoco/THIRD_PARTY_NOTICES.md), including model
assets preserved inside diagnostic archives.
