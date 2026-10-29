# Newton Robot Tasks

Build a small Newton simulation, migrate a MuJoCo task that picks up both cubes and places them in a receiving box, and then use a robot gripper to place two cubes, a cloth patch and a cable into a box. The coupled task assigns the robot and cubes to MuJoCo, and the cloth and cable to Vertex Block Descent (VBD). Newton manages the shared scene, state, controls and contact interfaces. On CUDA, MuJoCo Warp advances the rigid solver.

This tutorial is self-contained. It includes the earlier MuJoCo/MJWarp **Python baselines** for comparison, so no other Hub tutorial or pull request is required. SO-101 and Seeed reBot DevArm share the learning path.

## Start here

Start with the complete two-cube example: **set up the environment below, run the box task, then open notebook 01 to learn the API**. The advanced cloth/cable lesson and performance sweep come later; neither is required for the first run.

Here is the advanced task you can build toward. This recorded SO-101 sequence includes the cubes, cable and cloth from lesson 03:

![Recorded SO-101 gripper sequence for the advanced cloth-and-cable lesson](notebooks/newton/assets/clean-table-so101.gif)

## Set up once

Use **Python 3.12**, [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. Clone the official repository once, then enter the tutorial:

```bash
git clone https://github.com/NVIDIA/accelerated-computing-hub.git
cd accelerated-computing-hub
```

If you already have a checkout, start at its root. Create this tutorial's environment:

```bash
cd tutorials/newton
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r notebooks/newton/requirements.lock.txt
source .venv/bin/activate
export PXR_WORK_THREAD_LIMIT=1
```

Keep this terminal in `tutorials/newton` for the commands below. Set `PXR_WORK_THREAD_LIMIT=1` before launching Python or Jupyter; restart existing kernels after changing it.

The pinned stack uses Newton 1.6.0, Warp 1.17.0, MuJoCo/MuJoCo Warp 3.12.0, OpenUSD 26.3, NumPy 2.5.3 and SciPy 1.18.1. Linux x86-64 requires glibc 2.35 or newer for the included Open3D wheel. CPU execution also works on macOS ARM64; CUDA requires a supported NVIDIA GPU.

## Run the box task

Run the complete SO-101 reference first. On a local desktop, open its viewer:

```bash
python notebooks/newton/part3/box_newton_demo.py --robot so101 --device cpu --viewer gl --report .generated/box-so101-viewer.json
```

On a remote machine or without a display, use the same task headlessly. The second command selects the reBot arm:

```bash
python notebooks/newton/part3/box_newton_demo.py --robot so101 --device cpu --viewer null --report .generated/box-so101-cpu.json
python notebooks/newton/part3/box_newton_demo.py --robot rebot --device cpu --viewer null --report .generated/box-rebot-cpu.json
```

Each command picks up both cubes and places them in the receiving box over 40 simulated seconds, then checks physical completion. A successful run prints `PASS` after both cubes are placed and all 40,000 integrations complete. Use a fresh report filename for every run; existing reports and recordings are preserved. The windowed command requires a working desktop/OpenGL display; the headless commands work without one.

On a supported NVIDIA GPU, run the same reference with an explicit CUDA device:

```bash
python notebooks/newton/part3/box_newton_demo.py --robot so101 --device cuda:0 --viewer null --report .generated/box-so101-cuda0.json
```

`box_newton_demo.py` is complete and runnable. `box_newton_exercise.py` contains the matching unfinished substep exercise. The optional earlier stacking example uses `solutions/so101_newton_solution.py`; the advanced cloth-and-cable example uses `solutions/clean_the_table_solution.py`.

## Learn with notebooks (optional)

With the environment active, register its kernel once and launch Jupyter:

```bash
python -m ipykernel install --user --name hub-newton --display-name "Hub Newton (Python 3.12)"
jupyter lab --notebook-dir notebooks
```

Open `newton/part3/` and select **Hub Newton (Python 3.12)** manually. Run each notebook from its first code cell to the last. Opening it from that folder gives the teaching notebooks their required working directory. Notebook metadata uses the Hub's shared Python/T4 template; select the registered kernel to use this tutorial's dependencies.

| Order | Notebook | What you do |
|---|---|---|
| 1 | [Newton fundamentals](notebooks/newton/part3/01__newton_fundamentals.ipynb) | Build, collide and step a scene; choose a solver; replicate worlds |
| 2 | [MuJoCo to Newton](notebooks/newton/part3/02__mujoco_to_newton.ipynb) | Run the complete two-cube box task, then implement its substep exercise |
| 3 | [Clean the table](notebooks/newton/part3/03__clean_the_table.ipynb) | Add cloth, cable and coupled solvers; allow tens of minutes for a full CPU rollout |
| 4 | [Final check](notebooks/newton/part3/04__final_check.ipynb) | Check both CPU box tasks; enable the separate stacking and coupled checks when needed |
| Optional | [CPU/GPU benchmark](notebooks/newton/part3/05__migration_benchmark.ipynb) | Run preflight; explicitly opt into measurements |

The teaching notebooks default to `REFERENCE = True`. Notebook 01 is complete inline; later notebooks select the supplied Python references. After completing the external files marked `TODO Step` (or the box substep exercise), set `REFERENCE = False`, restart the kernel and rerun. There are no separate solution notebooks, and you do not need to run every Python file first.

Notebooks 02 and 04 run the full box task on CPU for both robots; CUDA is opt-in. In notebook 04, `RUN_LEGACY_AND_COUPLED = False` leaves the stacking and cloth/cable checks disabled. Notebook 05 defaults to `RUN_BENCHMARK = False` and runs preflight only. Notebook 03 is the slower coupled-material lesson: full CPU rollouts can take tens of minutes per robot. Long cells capture subprocess output until completion.

## Docker and Brev

The [Compose configuration](brev/docker-compose.yml) supplies the repository's standard `base` and `jupyter` services. It installs the same hash-locked Python 3.12 environment and sets the OpenUSD launch variable. The CUDA 13.1 image needs a compatible host driver and NVIDIA Container Toolkit; the services expose JupyterLab on port 8888. Compose reserves 4 GiB of shared memory for full CPU histories and independent validation; this is separate from GPU memory.

From the Hub repository root:

```bash
brev/dev-build.bash newton
brev/dev-start.bash newton
brev/dev-test.bash newton
```

The default test entrypoint runs the regression suite and notebooks 01, 02, 04 and the preflight-only 05. To include the long coupled notebook 03, set `NEWTON_NOTEBOOKS_INCLUDE_FINAL=1` before starting Compose, or run `NEWTON_NOTEBOOKS_INCLUDE_FINAL=1 bash brev/test.bash` inside this tutorial's environment. `brev/test.bash 01` through `05` select individual notebooks; `03` explicitly opts into the coupled task; standard pytest paths and flags are also accepted.

## Compare CPU and GPU performance

The [benchmark notebook](notebooks/newton/part3/05__migration_benchmark.ipynb) explains the timing boundaries, physical checks and full sweep. It runs the same two-cube task through Newton on persistent CPU workers and one replicated CUDA model. Run preflight before collecting measurements:

```bash
python notebooks/newton/part3/migration_benchmark.py --scope newton-batch --preset developer --device cuda:0 --cpu-threads 32 --nconmax 64 --njmax 128 --timeout 7200 --preflight
```

For a small exploratory comparison, use a fresh output directory:

```bash
python notebooks/newton/part3/migration_benchmark.py --scope newton-batch --robot so101 --worlds 1 16 --device cuda:0 --cpu-threads 32 --nconmax 64 --njmax 128 --repeats 5 --warmups 1 --timeout 7200 --output-dir .generated/benchmark-so101
```

Each world completes 40 simulated seconds and all 40,000 integrations. Every warm-up and measured episode must pass before a timing is accepted. Reports separate simulation, output collection and validation; retain failed cases and compare matching successful configurations. A small world list is an exploratory subset. `--cpu-only` runs the CPU baseline without measuring GPU performance. Each invocation uses one GPU.

Record the source commit, hardware and environment with your results. The [9 October 2026 study](https://github.com/NVIDIA/accelerated-computing-hub/blob/5b97e56a1fd832e1b2428479d440eccb4b8f8403/tutorials/newton/notebooks/newton/benchmark-results/2026-10-09-validated-box-gpu0/README.md) remains available at its historical revision for comparison.

## Add cloth and a cable

After notebook 02, follow notebook 03 to manipulate four objects sequentially. `SolverMuJoCo` owns the robot and cubes, `SolverVBD` owns the cloth and cable, and `SolverCoupledProxy` exchanges contact reactions at their interfaces. VBD and coupling are experimental in Newton 1.6.

Run the completed SO-101 sequence headlessly and save its states:

```bash
python notebooks/newton/part3/solutions/clean_the_table_solution.py --robot so101 --device cpu --viewer null --test --report .generated/clean-so101.json --record .generated/clean-so101.npz
```

Use `--viewer gl` for a local viewer or `--device cuda:0` for CUDA. The checks require each object to be gripped, lifted, carried, released and settled inside the box, with the gripper clear at the end. The cloth is a single-layer shirt silhouette and the cable is a rod approximation; this scripted lesson is not a general cleanup policy.

Render a passing recording without advancing physics:

```bash
python notebooks/newton/tools/render_clean_table.py --report .generated/clean-so101.json --record .generated/clean-so101.npz --output-dir .generated/clean-so101-figures --gif
```

The renderer checks that the report and recording match. Add `--renderer gl` to use the official robot meshes instead of the default collision geometry.

## Further reading and attribution

- [State of Simulation for Physical AI](https://huggingface.co/blog/nvidia/state-of-simulation-for-physical-ai)
- [How to Use NVIDIA Warp and MuJoCo Warp](https://huggingface.co/blog/nvidia/how-to-use-nvidia-warp-and-mjwarp)
- [Newton source repository](https://github.com/newton-physics/newton)
- [NVIDIA learning path: Newton fundamentals](https://docs.nvidia.com/learning/physical-ai/getting-started-with-newton/latest/newton-fundamentals/overview.html)
- [Newton documentation](https://newton-physics.github.io/newton/stable/) and [solver coupling](https://newton-physics.github.io/newton/stable/concepts/coupling.html)
- [Third-party notices](notebooks/newton/THIRD_PARTY_NOTICES.md) and [retained licenses](notebooks/LICENSES/)

Robot models download from pinned upstream revisions. Automatic caches require unchanged tracked robot files; use a fresh `NEWTON_CACHE_PATH` for USD assets or `MUJOCO_MENAGERIE_CACHE` for MJCF assets if a cache has been modified. Use `--menagerie-path` only when intentionally selecting a different MJCF model. The companion code retains its MIT attribution to Johnny Nuñez Cano, including the MuJoCo/MJWarp comparison baselines.
