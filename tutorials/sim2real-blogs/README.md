# Sim2Real Blogs

Companion tutorials for the simulation-to-real robotics blog series. Article 2
runs the same receiving-box task on native MuJoCo CPU and MuJoCo Warp GPU:
each arm grasps both cubes, carries and releases them into the box, then withdraws.
The lessons check both SO-101 and Seeed reBot DevArm. They do not deploy a policy
to a physical robot.

**To run a working demo, use the scripts in `solutions/`.** The similarly named
scripts outside that folder are exercises with missing code. Notebooks are optional,
and you do not need any later article or another Hub tutorial.

| What you want to do | Guide |
|---|---|
| Run MuJoCo on CPU | [CPU demo](notebooks/mujoco/part1/README.md) |
| Run MJWarp on an NVIDIA GPU | [GPU demo](notebooks/mujoco/part2/README.md) |
| Work through the lessons | [Notebook order and kernel setup](notebooks/mujoco/README.md#learn-with-notebooks-optional) |
| Measure performance | [Benchmarking](notebooks/mujoco/BENCHMARK.md) |

## Quickstart (local)

Use Linux or macOS, Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
An NVIDIA GPU is needed only for the GPU lesson and measurements. From the directory
where you want to clone the Hub:

```bash
git clone https://github.com/NVIDIA/accelerated-computing-hub.git
cd accelerated-computing-hub/tutorials/sim2real-blogs/notebooks/mujoco
uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements.lock.txt
source .venv/bin/activate
python ../../doctor.py
python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20
```

Keep this working directory for all commands in the CPU and GPU lesson guides.
In each new terminal, return here and run `source .venv/bin/activate`.
The last command opens a viewer, downloads the robot model on first use, and closes
after 40 simulated seconds. Success ends with
`Task complete: both cubes are in the receiving box (verified).`
Add `--headless-steps 2000` without a display, or `--robot rebot` for the second arm.
Esc, Q or Ctrl+C stop the viewer early. macOS viewer execution has not been validated.

Then, on a CUDA machine, run the same task on the GPU. **This one-world check can be slower than native MuJoCo on CPU even after warm-up.** See the [GPU guide](notebooks/mujoco/part2/README.md#run-the-completed-demo) for startup expectations and the separate batched measurements:

```bash
python part2/solutions/so101_mjwarp_solution.py --task box --sim-substeps 20 --device cuda:0
```

The first GPU run compiles kernels and can take several minutes.
See the [file guide](notebooks/mujoco/README.md#which-files-do-i-run) for exercises,
answer snippets and helper modules, or follow the [optional notebook path](notebooks/mujoco/README.md#learn-with-notebooks-optional).

No environment variable is needed for the robot models. A valid cache from an
earlier run is reused; anything else in `~/.cache/mujoco_menagerie` is left alone
or moved aside, and a verified copy is downloaded into a folder named after the
pinned commit. Offline machines: copy `~/.cache/mujoco_menagerie` and
`~/.cache/mujoco_menagerie_rebot` from a machine where the lessons already ran.

With conda instead of uv: `conda create -y -n sim2real python=3.12 git`,
`conda activate sim2real`, then
`pip install --require-hashes -r requirements.lock.txt` and `python ../../doctor.py`,
from the same `notebooks/mujoco` directory.

## Local and hosted setup

Use Python 3.12 with the module's hash lock. This tutorial uses the measured MuJoCo 3.8.0 / MuJoCo Warp 3.8.0.3 / Warp 1.15.0 line.

The existing Brev Docker Compose services and shared Hub entrypoint are retained.
The image installs the same physics lock before adding notebook/UI tools under
constraints. From the **Hub repository root**, build and test it through the Hub scripts:

```bash
brev/dev-build.bash sim2real-blogs
brev/dev-test.bash sim2real-blogs
```

A container build, Brev deployment or Colab run is separate validation; imported
workstation measurements do not establish those results. The Colab links open the notebooks from the Hub main branch; setup downloads the
runtime source from NVIDIA’s official Hub repository. `main` is resolved once to a full commit; set `HUB_SOURCE_REF` to a full commit to reproduce a run. Setup prints that Git SHA,
refuses to overwrite a checkout at another revision, and skips LFS downloads for
teaching code. Teaching cells need a Python 3.12 kernel; the benchmark notebook
uses an isolated Python 3.12 environment for hosted subprocesses.

### Hosted notebooks

Open notebooks 01, 02 and 03 in order; notebook 04 is an optional benchmark.
For local Jupyter, use the [local kernel instructions](notebooks/mujoco/README.md#learn-with-notebooks-optional).

| Notebook | Description | Colab |
|---|---|---|
| [MuJoCo fundamentals](notebooks/mujoco/part1/01__mujoco_fundamentals.ipynb) | Inspect a model, state and CPU stepping. | [Open](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/sim2real-blogs/notebooks/mujoco/part1/01__mujoco_fundamentals.ipynb) |
| [Pick and place](notebooks/mujoco/part1/02__pick_and_place.ipynb) | Both robots place both cubes in the box. | [Open](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/sim2real-blogs/notebooks/mujoco/part1/02__pick_and_place.ipynb) |
| [MuJoCo Warp](notebooks/mujoco/part2/03__mujoco_warp.ipynb) | The same task on one selected GPU. | [Open](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/sim2real-blogs/notebooks/mujoco/part2/03__mujoco_warp.ipynb) |
| [CPU/GPU benchmark](notebooks/mujoco/part2/04__cpu_gpu_benchmark.ipynb) | Preflight first; optionally measure 1 through 2048 environments. | [Open](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/sim2real-blogs/notebooks/mujoco/part2/04__cpu_gpu_benchmark.ipynb) |

## Evidence and attribution

See [validation scope](VALIDATION.md) for the checks performed on this tutorial.

[Migration provenance](notebooks/mujoco/MIGRATION_MANIFEST.json) records relocation
and source identities. The original result, notebook, viewer and reference packages
are retained byte-for-byte, including failed cases and their historical paths.
Pull the required LFS objects when inspecting archived evidence; notebook physics
does not require downloading those archives.

The imported materials retain their [MIT license](notebooks/mujoco/LICENSE) and
[third-party notices](notebooks/mujoco/THIRD_PARTY_NOTICES.md), including model
assets preserved inside diagnostic archives.
