# Newton Robot Tasks

Make an SO-101 arm pick up two cubes and place them in a box using Newton.
Then add cloth and a cable to learn how different physics solvers work together.
You can also run the tasks with Seeed's reBot DevArm.

## Start here

**Install the environment, run the box task, then open notebook 01** to work
through the code. Lesson 03 adds the cable and cloth shown here:

![Recorded SO-101 gripper sequence for the advanced cloth-and-cable lesson](notebooks/newton/assets/clean-table-so101.gif)

## Set up once

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/),
then clone the repository:

```bash
git clone https://github.com/NVIDIA/accelerated-computing-hub.git
cd accelerated-computing-hub
```

From the repository root, create a Python 3.12 environment:

```bash
cd tutorials/newton
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r notebooks/newton/requirements.lock.txt
source .venv/bin/activate
export PXR_WORK_THREAD_LIMIT=1
```

Run the commands below from `tutorials/newton` with `.venv` active.
Set `PXR_WORK_THREAD_LIMIT=1` before launching Python or Jupyter to avoid an
OpenUSD parser issue; restart any kernels that are already running.

This installs Newton 1.7.0, Warp 1.18.0, MuJoCo/MuJoCo Warp 3.15.0 and OpenUSD
26.3. Linux x86-64 needs glibc 2.35 or newer. CPU examples also run on macOS
ARM64. Warp 1.18 requires an NVIDIA Turing or newer GPU (compute capability
7.5+) and an R580 or newer driver for CUDA.

## Run the box task

On a local desktop, run SO-101 with a viewer:

```bash
python notebooks/newton/part3/box_newton_demo.py --robot so101 --device cpu --viewer gl --report .generated/box-so101-viewer.json
```

Without a display, use `--viewer null`. Choose either robot:

```bash
python notebooks/newton/part3/box_newton_demo.py --robot so101 --device cpu --viewer null --report .generated/box-so101-cpu.json
python notebooks/newton/part3/box_newton_demo.py --robot rebot --device cpu --viewer null --report .generated/box-rebot-cpu.json
```

After 40 simulated seconds, both cubes should be in the box and the gripper
clear. The program prints `PASS` when the task checks and all 40,000 physics
steps succeed. Use a new report filename each time.

Run the same task on GPU:

```bash
python notebooks/newton/part3/box_newton_demo.py --robot so101 --device cuda:0 --viewer null --report .generated/box-so101-cuda0.json
```

To implement the physics step yourself, edit `box_newton_exercise.py` after
running the demo. Notebook 02 walks through that exercise.

## Learn with notebooks (optional)

Register the environment as a Jupyter kernel, then open Jupyter:

```bash
python -m ipykernel install --user --name hub-newton --display-name "Hub Newton (Python 3.12)"
jupyter lab --notebook-dir notebooks
```

Open `newton/part3/`, select **Hub Newton (Python 3.12)** as the kernel, and
run each notebook from its first cell. Keep the notebooks in that folder so
their file paths resolve.

| Order | Notebook | What you do |
|---|---|---|
| 1 | [Newton fundamentals](notebooks/newton/part3/01__newton_fundamentals.ipynb) | Build, collide and step a scene; choose a solver; replicate worlds |
| 2 | [MuJoCo to Newton](notebooks/newton/part3/02__mujoco_to_newton.ipynb) | Run the complete two-cube box task, then implement its substep exercise |
| 3 | [Clean the table](notebooks/newton/part3/03__clean_the_table.ipynb) | Add cloth, cable and coupled solvers; allow tens of minutes for a full CPU rollout |
| 4 | [Final check](notebooks/newton/part3/04__final_check.ipynb) | Check both CPU box tasks; enable the separate stacking and coupled checks when needed |
| Optional | [CPU/GPU benchmark](notebooks/newton/part3/05__migration_benchmark.ipynb) | Run preflight; explicitly opt into measurements |

The notebooks run the supplied solutions with `REFERENCE = True`. Complete
the files marked `TODO Step`, then set `REFERENCE = False`, restart the kernel
and rerun to try your code.

Notebooks 02 and 04 start on CPU. Enable their GPU options to use CUDA.
In notebook 04, `RUN_LEGACY_AND_COUPLED = True` adds the stacking and cloth/cable
checks. Notebook 05 checks the configuration first; set `RUN_BENCHMARK = True`
to start measurements. Allow tens of minutes per robot for notebook 03 on CPU;
its long cells show output when the run finishes.

## Docker and Brev

The [container](brev/docker-compose.yml) includes the environment and JupyterLab
on port 8888. Its CUDA 13.1 image needs a compatible NVIDIA driver and NVIDIA
Container Toolkit. It reserves 4 GiB of shared memory for the CPU task histories.

From the Hub repository root:

```bash
brev/dev-build.bash newton
brev/dev-start.bash newton
brev/dev-test.bash newton
```

The test command runs the tests and notebooks 01, 02, 04 and notebook 05's setup
check. Set `NEWTON_NOTEBOOKS_INCLUDE_FINAL=1` before starting Compose to include
the longer notebook 03. To run one notebook from `tutorials/newton`, use
`bash brev/test.bash 01` through `05`.

## Compare CPU and GPU performance

Compare the full two-cube task on CPU workers and one GPU. First check the
configuration:

```bash
python notebooks/newton/part3/migration_benchmark.py --scope newton-batch --preset developer --device cuda:0 --cpu-threads 32 --nconmax 64 --njmax 128 --timeout 7200 --preflight
```

Then try batches of 1 and 16 worlds, using a new output directory:

```bash
python notebooks/newton/part3/migration_benchmark.py --scope newton-batch --robot so101 --worlds 1 16 --device cuda:0 --cpu-threads 32 --nconmax 64 --njmax 128 --repeats 5 --warmups 1 --timeout 7200 --output-dir .generated/benchmark-so101
```

Each world runs the complete 40-second task. A failed task is excluded from
the timings. Compare CPU and GPU at the same world count, including the time
to collect and check the results. Setup and warm-up are listed separately.
Use `--cpu-only` for the CPU baseline; GPU runs each use one device.

The [benchmark notebook](notebooks/newton/part3/05__migration_benchmark.ipynb)
walks through larger batches. For earlier measurements, see the
[9 October 2026 study](https://github.com/NVIDIA/accelerated-computing-hub/blob/5b97e56a1fd832e1b2428479d440eccb4b8f8403/tutorials/newton/notebooks/newton/benchmark-results/2026-10-09-validated-box-gpu0/README.md).

## Add cloth and a cable

Notebook 03 adds cloth and a cable. `SolverMuJoCo` handles the robot and cubes,
`SolverVBD` handles the cloth and cable, and `SolverCoupledProxy` passes contact
forces between them. VBD and coupling are experimental in Newton 1.7.

Run SO-101 without a viewer and save the motion:

```bash
python notebooks/newton/part3/solutions/clean_the_table_solution.py --robot so101 --device cpu --viewer null --test --report .generated/clean-so101.json --record .generated/clean-so101.npz
```

Use `--viewer gl` to watch locally or `--device cuda:0` to run on GPU.
The arm must grip, lift, carry and release each object, then move clear while
the object settles in the box. The shirt is a single layer of cloth and the
cable is modeled as a rod.

Turn a successful run into a GIF:

```bash
python notebooks/newton/tools/render_clean_table.py --report .generated/clean-so101.json --record .generated/clean-so101.npz --output-dir .generated/clean-so101-figures --gif
```

Add `--renderer gl` to show the robot meshes instead of collision geometry.

## Further reading and attribution

- [State of Simulation for Physical AI](https://huggingface.co/blog/nvidia/state-of-simulation-for-physical-ai)
- [How to Use NVIDIA Warp and MuJoCo Warp](https://huggingface.co/blog/nvidia/how-to-use-nvidia-warp-and-mjwarp)
- [Newton source repository](https://github.com/newton-physics/newton)
- [NVIDIA learning path: Newton fundamentals](https://docs.nvidia.com/learning/physical-ai/getting-started-with-newton/latest/newton-fundamentals/overview.html)
- [Newton documentation](https://newton-physics.github.io/newton/stable/) and [solver coupling](https://newton-physics.github.io/newton/stable/concepts/coupling.html)
- [Third-party notices](notebooks/newton/THIRD_PARTY_NOTICES.md) and [retained licenses](notebooks/LICENSES/)

Robot models download on first use. If you edited a cached model, choose a new
`NEWTON_CACHE_PATH` for USD or `MUJOCO_MENAGERIE_CACHE` for MJCF to download a
clean copy. Use `--menagerie-path` to load your own MJCF model.

The tutorial code, including its MuJoCo/MJWarp baselines, retains its MIT
attribution to Johnny Nuñez Cano.
