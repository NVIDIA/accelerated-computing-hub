# SO-101 simulation: from MuJoCo to MuJoCo Warp

These three lessons accompany the second article in the simulation series,
*How to Use NVIDIA Warp and MJWarp to Accelerate Robotics Simulation and Learning
Workflows*. Start with a MuJoCo CPU scene, complete an SO-101 pick-and-place
controller, then migrate the physics loop to MuJoCo Warp and measure batched
physics throughput.

| Lesson | Topic |
| --- | --- |
| [MuJoCo fundamentals](part1/05__mujoco_fundamentals.ipynb) | Compile MJCF, inspect model and state, control actuators, and render the robot. |
| [Pick and place](part1/06__pick_and_place.ipynb) | Complete a CPU simulation loop and check that the red cube is stacked on the blue cube. |
| [MuJoCo Warp](part2/07__mujoco_warp.ipynb) | Upload the compatible model, seed GPU state, validate one world, capture a CUDA graph, and measure a batch. |

The notebooks patch `TODO` sections in the exercise scripts. The complete,
runnable programs are in each part's `solutions/` directory; the small
`step_*.py` files contain individual exercise answers. Keep each notebook beside
its supporting scripts. Both parts include the same task and asset helpers so
each lesson can run from its own directory.

## Requirements and setup

Use Python 3.12 and Git. The CPU lessons run without a GPU; the MJWarp lesson
requires an NVIDIA CUDA GPU supported by
[Warp](https://nvidia.github.io/warp/installation.html). The pinned environment
uses MuJoCo 3.12.0, MuJoCo Warp 3.12.0, and Warp 1.17.0. It does not require
Newton. First use downloads the selected robot's pinned MuJoCo Menagerie assets
and preserves their licenses in the local cache.

From a fresh checkout:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse https://github.com/NVIDIA/accelerated-computing-hub.git
cd accelerated-computing-hub
git sparse-checkout set tutorials/warp
uv venv --python 3.12
uv pip install --python .venv/bin/python -r tutorials/warp/notebooks/mujoco/requirements.txt
source .venv/bin/activate
cd tutorials/warp/notebooks/mujoco
```

Run `jupyter lab` to work through the notebooks, or run the completed examples
below. On Colab, select a GPU runtime for MuJoCo Warp and run the notebook's setup
cell first. Brev users can use the existing [Warp environment](../../README.md).

## Validate the task

The task uses 50 control frames per second and 10 physics substeps per frame,
so both backends use a **0.002 s physics timestep**. The SO-101 arm actuator force
limits are increased to ±30 N·m for this demonstration; the gripper limit stays
unchanged. This is a simulation exercise, not validation of the stock robot's
hardware capabilities.

```bash
# CPU: 600 control frames, then assert that the stack succeeded.
python part1/solutions/so101_pick_place_solution.py --headless-steps 600 --test

# CUDA: validate the same task with one MJWarp world.
python part2/solutions/so101_mjwarp_solution.py --headless-steps 600 --test
```

`--test` rejects non-finite state and a failed stack. The 44 mm cubes must settle
with horizontal center error below 15 mm and vertical center separation within
10 mm of one cube edge. The check also requires the controller to finish and
the blue cube to remain on the table. CPU float64 and MJWarp float32 trajectories can differ;
passing these task checks does not establish bitwise or trajectory equivalence.
Capacity overflow invalidates a GPU run and must be fixed before reporting its
result.

Omit `--headless-steps` and `--test` to open the interactive viewer on a machine
with a display. The helper handles the `mjpython` launcher on macOS. The optional
`--robot rebot` profile uses the Seeed reBot DevArm; validate it separately before
reporting task results. Use `--menagerie-path /path/to/mujoco_menagerie` to supply
an existing model checkout, or set `MUJOCO_MENAGERIE_CACHE` to choose the download
cache location. A user-supplied checkout may differ from the pinned assets.
Managed downloads live under `<cache root>/<robot cache name>/<full commit>`;
the default cache root is `~/.cache`. Reusing a managed checkout requires its
Git revision to match the requested pin.

## Measure physics throughput

After the one-world CUDA task passes:

```bash
python part2/solutions/so101_mjwarp_solution.py --benchmark --nworld 64 --steps 200
python part2/scaling_study.py --worlds 1 16 64 256 --steps 200
```

These benchmarks replicate the initialized state and hold controls fixed. They
measure **physics stepping**, excluding controller updates, rendering, model
upload, and state transfers during the timed region. They do not measure
completed pick-and-place tasks or training speed. Compilation and warm-up are
excluded, and CUDA is synchronized around timing. Report milliseconds per
batched step, aggregate world-steps per second, world count, capacities, software
versions, and hardware together.

The notebook's CPU baseline advances independent worlds sequentially. Its
comparison does not represent all CPU parallelism. The standalone scaling tool
compares GPU throughput across batch sizes, relative to its first batch size.
For a parallel CPU comparison, MuJoCo also provides
[`mujoco.rollout`](https://mujoco.readthedocs.io/en/stable/python.html#rollout).
Throughput and memory use depend on the model, contacts, solver settings, and
hardware. Increase world count gradually and verify capacity before drawing
performance conclusions.

## Validation

From this directory, with the environment active:

```bash
uv pip install pytest
python -m pytest ../../test/test_mujoco.py -v
```

The suite exercises the CPU task and checks task failure detection. CUDA tests
skip when a CUDA device is unavailable. On an NVIDIA GPU, the suite also runs
the one-world task and a small captured benchmark. First use needs network
access for the pinned robot assets. The Warp Brev test entrypoint runs this
suite after checking GPU availability.

## Attribution

Adapted from [Johnny Nuñez Cano's Article 2 tutorials](https://github.com/johnnynunez/blogs/tree/2586ee1519bddbe66ea542e5177f550a71ee0a9e/Article_2).
The imported scripts and notebooks retain their [MIT license](LICENSE).
Robot assets are downloaded rather than redistributed; see
[third-party notices](THIRD_PARTY_NOTICES.md) for pinned commits, licenses, and
scene attribution.
