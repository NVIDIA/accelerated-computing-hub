# MuJoCo CPU → MJWarp GPU

Make SO-101 pick up two cubes and place them in a box. Run it on CPU first,
then on GPU, and compare how quickly each handles many copies of the scene.
Use `--robot rebot` to try the same task with Seeed's reBot arm.

## Setup

You need Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).
CPU examples work on Linux and macOS on Apple Silicon. The locked Warp 1.18 wheel needs an
NVIDIA Turing (`sm_75`) or newer GPU and an R580 or newer driver for CUDA.

```bash
git clone https://github.com/NVIDIA/accelerated-computing-hub.git
cd accelerated-computing-hub/tutorials/sim2real-blogs/notebooks/mujoco
uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements.lock.txt
source .venv/bin/activate
python ../../doctor.py
```

Run the commands below from `notebooks/mujoco` with `.venv` active.
Continue when the setup check prints `Ready: no blocking problems`.
This installs MuJoCo 3.17.0, MuJoCo Warp 3.17.0, Warp 1.18.0 and Pillow 13.0.0.
The robot models download on first use.

## Run the task

Run on CPU:

```bash
python part1/solutions/so101_pick_place_solution.py --task box --sim-substeps 20
```

Then run on GPU:

```bash
python part2/solutions/so101_mjwarp_solution.py --task box --sim-substeps 20 --device cuda:0
```

Each command opens a viewer. After 40 simulated seconds, both cubes should be
in the box and the gripper clear. The program checks this and prints
`Task complete: both cubes are in the receiving box (verified).`

Without a display, add `--headless-steps 2000`. To save the result, add
`--report .generated/box.json`. Space pauses the viewer; Esc, Q, window close
or Ctrl+C cancels the run.

The first GPU run takes longer while kernels compile. A single scene can also
run slower on GPU after that; the performance section below shows how to try
larger batches. Use `--device cuda:1` to select a second GPU.

For the shorter cube-stacking exercise, use `--task stack --headless-steps 600`.

## Notebooks

To work through the code step by step, install and open Jupyter:

```bash
uv pip install --python .venv/bin/python -c requirements.lock.txt jupyterlab
python -m ipykernel install --user --name hub-sim2real --display-name "Sim2Real (MuJoCo 3.17)"
jupyter lab
```

Select **Sim2Real (MuJoCo 3.17)** as the kernel and run the notebooks in order:

| Order | Notebook | Purpose |
|---|---|---|
| 1 | [MuJoCo fundamentals](notebooks/mujoco/part1/01__mujoco_fundamentals.ipynb) | Inspect and render the CPU scene. |
| 2 | [CPU pick and place](notebooks/mujoco/part1/02__pick_and_place.ipynb) | Run the box task and complete the CPU exercise. |
| 3 | [MJWarp](notebooks/mujoco/part2/03__mujoco_warp.ipynb) | Move the task to CUDA. |
| 4, optional | [CPU/GPU benchmark](notebooks/mujoco/part2/04__cpu_gpu_benchmark.ipynb) | Compare complete workloads at different batch sizes. |

Notebooks 2 and 3 run the completed solutions with `REFERENCE = True`.
To try the exercises, fill the `TODO Step` markers in `part1/so101_pick_place.py`
or `part2/so101_mjwarp.py`, then set `REFERENCE = False`.

## Measure performance

First measure physics stepping at several batch sizes:

```bash
python part2/scaling_study.py --worlds 1 64 1024 2048 --steps 200 --device cuda:0
```

`world-steps/s` counts simulation steps across all worlds; `ms/step` is the
time to advance the whole batch. The CPU baseline uses one core. Controls stay
fixed during this test, so it measures physics speed rather than completed tasks.

To time the complete box task, check the configuration and then run the sweep:

```bash
python part2/migration_benchmark.py --task box --preset workstation --device cuda:0 --cpu-threads 32 --nconmax 64 --njmax 128 --preflight
python part2/migration_benchmark.py --task box --cpu-baseline pool --robot so101 \
  --worlds 1 16 32 64 128 256 512 1024 2048 --cpu-threads 32 --device cuda:0 \
  --nconmax 64 --njmax 128 --repeats 5 --warmups 1 --order-seed 0 \
  --timeout 7200 --max-trajectory-mib 8192 --output-dir .generated/box-so101
```

Start with fewer `--worlds` and CPU workers if needed. Use `--robot rebot` and
a new output directory for the other arm. In the benchmark notebook, set
`RUN_BENCHMARK = True` when ready to measure.

CPU and GPU replay the same controls for 40 simulated seconds. Both cubes must
be grasped, carried, released and settled in the box before the gripper withdraws.
A failed task or simulation error excludes that configuration from the timings.

Compare CPU and GPU at the same world count. The report lists simulation,
output transfer and task-check time separately from setup and warm-up.
At 2,048 worlds, observation history alone uses about 953 MiB, plus memory for
the solver and host copies. Each run uses one GPU.

## Hosted environments

On Colab, select a GPU runtime for the MJWarp lessons and run the notebook from
the first cell. Check the assigned hardware before starting a benchmark and
download your results before closing the session.

For the Hub's Brev container, run from the **repository root**:

```bash
brev/dev-build.bash sim2real-blogs
brev/dev-test.bash sim2real-blogs
```

To run the tests locally, use `python -m pytest ../../test` from
`notebooks/mujoco` with `.venv` active.

## License

See [LICENSE](notebooks/mujoco/LICENSE) and
[third-party notices](notebooks/mujoco/THIRD_PARTY_NOTICES.md) for source and model
attribution.
