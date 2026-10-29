# MuJoCo CPU → MJWarp GPU

Run a robot task on CPU, move the same task to one CUDA GPU, then measure how
batching changes throughput. SO-101 grasps two cubes, places them in a box and
withdraws its gripper. The optional `--robot rebot` profile uses a second arm.

Start with the completed scripts in `solutions/`. The matching scripts outside
that directory are exercises; `step_NN_*.py` files contain individual answers.
Shared scene, controller and task helpers live in `sim2real_shared/`.

## Setup

You need Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
CPU examples work on Linux and macOS. GPU examples need an NVIDIA CUDA GPU.

```bash
git clone https://github.com/NVIDIA/accelerated-computing-hub.git
cd accelerated-computing-hub/tutorials/sim2real-blogs/notebooks/mujoco
uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements.lock.txt
source .venv/bin/activate
python ../../doctor.py
```

Keep `notebooks/mujoco` as your working directory for the commands below.
In a new terminal, return there and activate `.venv` again. The doctor should end
with `Ready: no blocking problems`; otherwise follow its printed fix.
The lock provides MuJoCo 3.8.0, MuJoCo Warp 3.8.0.3, Warp 1.15.0 and Pillow 13.0.0.
Robot models download automatically at pinned revisions on first use.

## Run the task

First run the CPU solution:

```bash
python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20
```

Then run the same task on one GPU:

```bash
python part2/solutions/so101_mjwarp_solution.py --task box --sim-substeps 20 --device cuda:0
```

Each command opens a viewer and runs 40 simulated seconds. A successful run prints
`Task complete: both cubes are in the receiving box (verified).`
Add `--headless-steps 2000` without a display, `--robot rebot` for the other arm,
or `--report .generated/box.json` to save the task result. Space pauses the viewer;
Esc, Q, window close or Ctrl+C stops it early. A cancelled run is incomplete.

The first GPU run compiles kernels. One GPU world can remain slower than CPU
after warm-up: this run checks the task, while batching measures GPU throughput.
Use `--device cuda:1` to select a second visible GPU; each run uses one device.

The earlier stacking exercise is available with `--task stack` and
`--headless-steps 600`. It has a different task and duration from the box example.

## Notebooks

For guided lessons, install Jupyter in the active environment and register it:

```bash
uv pip install --python .venv/bin/python -c requirements.lock.txt jupyterlab
python -m ipykernel install --user --name hub-sim2real --display-name "Sim2Real (MuJoCo 3.8)"
jupyter lab
```

Select **Sim2Real (MuJoCo 3.8)** as the kernel and run the notebooks in order:

| Order | Notebook | Purpose |
|---|---|---|
| 1 | [MuJoCo fundamentals](notebooks/mujoco/part1/01__mujoco_fundamentals.ipynb) | Inspect and render the CPU scene. |
| 2 | [CPU pick and place](notebooks/mujoco/part1/02__pick_and_place.ipynb) | Run the box task and complete the CPU exercise. |
| 3 | [MJWarp](notebooks/mujoco/part2/03__mujoco_warp.ipynb) | Move the task to CUDA. |
| 4, optional | [CPU/GPU benchmark](notebooks/mujoco/part2/04__cpu_gpu_benchmark.ipynb) | Compare complete workloads at different batch sizes. |

Notebooks 2 and 3 start with `REFERENCE = True` and run the completed solution
for both robots. Fill the `TODO Step` markers in `part1/so101_pick_place.py` or
`part2/so101_mjwarp.py`, then select `REFERENCE = False` to run your exercise.
The notebooks leave your exercise files intact.

## Measure performance

For a quick check of batched physics with fixed controls:

```bash
python part2/scaling_study.py --worlds 1 64 1024 2048 --steps 200 --device cuda:0
```

Read `world-steps/s` as aggregate throughput and `ms/step` as the time to advance
the whole batch. Its CPU baseline uses one core. This measurement holds the
robot at its initial controls; it does not complete the pick-and-place task.

For the complete box task, run preflight before the optional full sweep:

```bash
python part2/migration_benchmark.py --task box --preset workstation --device cuda:0 --cpu-threads 32 --nconmax 64 --njmax 128 --preflight
python part2/migration_benchmark.py --task box --cpu-baseline pool --robot so101 \
  --worlds 1 16 32 64 128 256 512 1024 2048 --cpu-threads 32 --device cuda:0 \
  --nconmax 64 --njmax 128 --repeats 5 --warmups 1 --order-seed 0 \
  --timeout 7200 --max-trajectory-mib 8192 --output-dir .generated/box-so101
```

Choose a smaller `--worlds` list and CPU worker count for a first run. Repeat with
`--robot rebot` and a new output directory to measure the other arm. The benchmark
notebook exposes the same options and defaults to preflight without measurements.

Both backends replay the same controls for 2,000 frames at 50 Hz, with 20 physics
substeps per frame. Each world must physically grasp, lift, carry, release and
settle both cubes inside the box, then withdraw the gripper. Incomplete tasks,
non-finite state and capacity overflow invalidate a configuration.

Reports separate simulation, output collection and validation time from setup,
compilation and warm-up. Compare equal world counts and complete task results;
failed configurations have no accepted timing. Preserve the commit, package lock,
hardware, CPU worker count and individual repetitions with any reported result.
At 2,048 worlds, observation history alone uses about 953 MiB; solver and host
copies need additional memory. Each invocation uses one GPU.

## Hosted environments

Colab setup in each notebook uses NVIDIA's official Hub repository. It resolves
`main` once and prints the full commit; set `HUB_SOURCE_REF` to a full commit to
reproduce a run. GPU notebooks require a CUDA runtime. Hosted hardware varies,
so run preflight before enabling a benchmark and download reports before exit.

For the Hub's Brev container, run from the **repository root**:

```bash
brev/dev-build.bash sim2real-blogs
brev/dev-test.bash sim2real-blogs
```

The container installs the same Python 3.12 lock. For local regression checks,
run `python -m pytest ../../test` from the activated `notebooks/mujoco` environment.

## License

See [LICENSE](notebooks/mujoco/LICENSE) and
[third-party notices](notebooks/mujoco/THIRD_PARTY_NOTICES.md) for source and model
attribution.
