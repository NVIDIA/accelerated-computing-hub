# Article 2 — MuJoCo on CPU and MJWarp on GPU

Companion code for *[How to Use NVIDIA Warp and MJWarp to Accelerate Robotics Simulation and Learning Workflows](https://huggingface.co/blog/nvidia/how-to-use-nvidia-warp-and-mjwarp)*.

Start with the completed CPU example, then run the same task on the GPU. **Use the scripts in `solutions/` to run a working example.** The similarly named scripts outside `solutions/` are exercises with missing code. Notebooks are optional.

Article 2 is self-contained: you do not need Newton or any later article.

| What you want to do | Guide |
| --- | --- |
| Run the robot on CPU | [Part 1: MuJoCo](part1/README.md) |
| Run the same task on an NVIDIA GPU | [Part 2: MJWarp](part2/README.md) |
| Work through the lessons | [Notebook order](#learn-with-notebooks-optional) |
| Measure CPU/GPU performance | [Benchmarking](BENCHMARK.md) |

## Set up once

Run the commands on these lesson pages from
**`accelerated-computing-hub/tutorials/sim2real-blogs/notebooks/mujoco`**.
Keep that working directory when moving between the CPU and GPU guides.

You need Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
If you have not cloned the Hub, start with the [local Quickstart](../../README.md#quickstart-local).
CPU examples run on Linux or macOS; GPU examples require an NVIDIA GPU with CUDA.
macOS viewer runs have not been validated.

```bash
uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements.lock.txt
source .venv/bin/activate
python ../../doctor.py
```

Expected final setup message: `Ready: no blocking problems`. If a check fails,
follow its printed `fix:` command. In each new terminal, return to this directory
and run `source .venv/bin/activate` again.

The lock installs MuJoCo 3.8.0, MuJoCo Warp 3.8.0.3 and Warp 1.15.0 with Python 3.12.
These are the versions used by this Hub tutorial and its retained measurements.

## Run a completed example

First, run on CPU:

```bash
python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20
```

Then, on a machine with an NVIDIA GPU, run the same task below. **This one-world check can be slower than native MuJoCo on CPU even after warm-up.** MJWarp targets throughput across many parallel worlds; see the [GPU guide](part2/README.md#run-the-completed-demo) before interpreting its speed:

```bash
python part2/solutions/so101_mjwarp_solution.py --task box --sim-substeps 20 --device cuda:0
```

Each command opens a viewer. The robot places the red and blue cubes in a box, and the window closes after 40 simulated seconds. A successful run ends with `Task complete: both cubes are in the receiving box (verified).` Robot models download automatically on first use; the first GPU run also compiles kernels.

Add `--headless-steps 2000` to run without a window, or `--robot rebot` to use the second robot. See the [CPU guide](part1/README.md) and [GPU guide](part2/README.md) for complete commands.

## Which files do I run?

| Files | Purpose | What to do |
| --- | --- | --- |
| `part1/solutions/so101_pick_place_solution.py`, `part2/solutions/so101_mjwarp_solution.py` | Completed CPU and GPU examples | Run these for the demo |
| `part1/so101_pick_place.py`, `part2/so101_mjwarp.py` | Exercises containing `TODO Step` markers | Complete the missing code before running them |
| `solutions/step_NN_*.py` in either part | Answer snippets for individual exercise steps | Read them as needed; they are not a sequence of programs to execute |
| `*.ipynb` | Guided lessons | Follow the optional order below |
| Other Python modules, such as `robots.py`, `utils.py` and `box_task.py` | Helpers used by the examples | The examples import them automatically |
| `part2/scaling_study.py`, `part2/migration_benchmark.py` | Optional performance measurements | Follow the [benchmark guide](BENCHMARK.md) |

You do not need to run every file in a folder.

## Learn with notebooks (optional)

Install Jupyter and register the kernel once, with the Article 2 environment active:

```bash
uv pip install --python .venv/bin/python -c requirements.lock.txt jupyterlab
python -m ipykernel install --user --name hub-sim2real --display-name "Sim2Real (MuJoCo 3.8)"
jupyter lab
```

Open the notebooks in this order. In each notebook, select **Sim2Real (MuJoCo 3.8)** under *Kernel > Change Kernel*, then run cells from top to bottom, starting with the setup cell.

| Order | Notebook | What it does |
| --- | --- | --- |
| 1 | [MuJoCo fundamentals](part1/01__mujoco_fundamentals.ipynb) | Introduces the CPU model, state and physics step |
| 2 | [CPU pick and place](part1/02__pick_and_place.ipynb) | Runs the completed two-cube task and guides the CPU exercise |
| 3 | [MJWarp](part2/03__mujoco_warp.ipynb) | Runs the same task on GPU and guides the GPU exercise; requires CUDA |
| 4, optional | [CPU/GPU benchmark](part2/04__cpu_gpu_benchmark.ipynb) | Checks the measurement setup, then lets you opt in to benchmarks |

The CPU pick-and-place and MJWarp notebooks start with `REFERENCE = True`, which runs the completed solution headlessly for both robots. Use the terminal commands above to open a local viewer. After filling in the `TODO Step` markers in your exercise script, set `REFERENCE = False` and rerun the selector and task cells to try your implementation. The notebooks do not overwrite your exercise files.

For Colab and Brev, follow the [hosted setup instructions](../../README.md#local-and-hosted-setup); local kernel registration is not needed there.

The benchmark notebook starts with `RUN_BENCHMARK = False`. **Run All checks the setup without collecting timings.** Follow its instructions to enable measurements when ready; you can skip that notebook entirely when learning the demos.

## Viewer controls and results

| Input | Effect |
| --- | --- |
| Space | Pause or resume |
| Esc, Q or window close | Stop early; the task completion check is skipped |
| Ctrl+C in the terminal | Interrupt the run |

`Task complete` means the measured task check passed. `Task FAILED` exits with code 1; an interrupted run is not a completed task. On macOS, use ordinary `python` for these scripts.

Models: [SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) and [reBot DevArm](https://github.com/google-deepmind/mujoco_menagerie/tree/main/seeed_rebot_devarm). See [third-party notices](THIRD_PARTY_NOTICES.md) for model licenses and attribution.

<a id="reading-the-benchmark-numbers"></a>

For benchmark units and historical scaling tables, see [Reading the benchmark numbers](BENCHMARK.md#reading-the-benchmark-numbers).

## Validation and attribution

The [migration record](MIGRATION_MANIFEST.json) records the imported source and evidence.
Archived packages retain their original paths and hashes. See the [benchmark guide](BENCHMARK.md)
for historical measurements, the [verification record](VERIFICATION.md),
[notebook execution record](notebook-validation/2026-10-10/README.md),
[viewer record](VIEWER_VALIDATION.md), and [13 October validation](validation/viewer-exit-and-benchmark-2026-10-13/README.md)
for the scope of earlier checks. These are not new Hub, Brev or Colab measurements.

The imported tutorial retains [Johnny Nuñez Cano’s MIT license](LICENSE).
Both original source notices are preserved in [LICENSES](LICENSES):
[Johnny Nuñez Cano](LICENSES/Johnny-Nunez-Cano-MIT.txt) and
[NVIDIA-dev](LICENSES/NVIDIA-dev-MIT.txt). Robot assets in diagnostic archives
retain their upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
