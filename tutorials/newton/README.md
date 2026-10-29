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

The pinned stack uses Newton 1.6.0, Warp 1.17.0, MuJoCo/MuJoCo Warp 3.12.0, OpenUSD 26.3, NumPy 2.5.3 and SciPy 1.18.1. Linux x86-64 requires glibc 2.35 or newer for the included Open3D wheel. CPU execution also works on macOS ARM64; CUDA requires a supported NVIDIA GPU. See the [setup details](notebooks/newton/README.md#setup) for platform and cache notes.

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

`box_newton_demo.py` is complete and runnable. `box_newton_exercise.py` contains the matching unfinished substep exercise. The optional earlier stacking example uses `solutions/so101_newton_solution.py`; the advanced cloth-and-cable example uses `solutions/clean_the_table_solution.py`. See the [file guide](notebooks/newton/README.md#which-files-do-i-run) before editing or running other scripts.

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

Notebooks 02 and 04 run the full box task on CPU for both robots; CUDA is opt-in. In notebook 04, `RUN_LEGACY_AND_COUPLED = False` leaves the stacking and cloth/cable checks disabled. Notebook 05 defaults to `RUN_BENCHMARK = False` and runs preflight only. Notebook 03 is the slower coupled-material lesson: historical CPU rollouts took approximately 26–29 minutes per robot on the recorded workstation. Long cells capture subprocess output until completion.

## Docker and Brev

The [Compose configuration](brev/docker-compose.yml) supplies the repository's standard `base` and `jupyter` services. It installs the same hash-locked Python 3.12 environment and sets the OpenUSD launch variable. The CUDA 13.1 image needs a compatible host driver and NVIDIA Container Toolkit; the services expose JupyterLab on port 8888. Compose reserves 4 GiB of shared memory for full CPU histories and independent validation; this is separate from GPU memory.

From the Hub repository root:

```bash
brev/dev-build.bash newton
brev/dev-start.bash newton
brev/dev-test.bash newton
```

The default test entrypoint runs the regression suite and notebooks 01, 02, 04 and the preflight-only 05. To include the long coupled notebook 03, set `NEWTON_NOTEBOOKS_INCLUDE_FINAL=1` before starting Compose, or run `NEWTON_NOTEBOOKS_INCLUDE_FINAL=1 bash brev/test.bash` inside this tutorial's environment. `brev/test.bash 01` through `05` select individual notebooks; `03` explicitly opts into the coupled task; standard pytest paths and flags are also accepted.

The updated image passed 274 regression tests across the initial suite and corrected stale-test rerun, three actual notebook checks, and complete CUDA box tasks for both robots. The [validation record](hub-validation/2026-10-10/README.md) retains the original failure and both snapshots. A live Brev deployment and Colab have not been validated. The original full physics validation and the relocation/container checks are distinguished in [VERIFICATION.md](notebooks/newton/VERIFICATION.md).

## Matched CPU/GPU comparison

The [benchmark guide](notebooks/newton/BENCHMARK.md) compares Newton `SolverMuJoCo` on native CPU and one CUDA device for the same two-cube box task. Each environment completes 2,000 frames at 50 Hz with 20 one-millisecond substeps: 40,000 integrations. Both robots use the same counts: 1, 16, 32, 64, 128, 256, 512, 1024 and 2048. The CPU uses up to 32 worker processes, capped by world count.

The retained workstation study accepted **35 of 36 configurations**; reBot GPU at 1024 worlds remains excluded with its original failed settling evidence. No accepted timing or speedup is reported for that configuration. This is a community measurement on documented hardware, not an official product benchmark. The [raw result package](notebooks/newton/benchmark-results/2026-10-09-validated-box-gpu0/README.md), [notebook evidence](notebooks/newton/notebook-validation/2026-10-10/README.md) and [viewer scope](notebooks/newton/VIEWER_VALIDATION.md) retain their separate provenance. ALOHA replay is a [separate reference workload](notebooks/newton/reference-benchmark/README.md).

The [current relocation map](notebooks/newton/current-relocation-manifest.json) binds imported files to their source bytes. The original relocation map remains historical. Archived source, logs, result JSON, compressed notebooks and archive manifests are preserved unchanged; their original paths describe the recorded run.

## Recorded results

These figures replay saved simulation states, using the official robot meshes. They show each payload carried by the gripper; final acceptance also checks release, containment, detached settling and jaw clearance.

![SO-101 carrying both cubes, cable and cloth during a recorded sequence](notebooks/newton/assets/clean-table-so101.png)

![reBot carrying both cubes, cable and cloth during a recorded sequence](notebooks/newton/assets/clean-table-rebot.png)

The [assets directory](notebooks/newton/assets/) contains both GIFs and the matching JSON/NPZ pairs. The result manifest binds the recordings and exported images to the original validated source. VBD and solver coupling remain experimental in Newton 1.6; this scripted setup is a learning exercise rather than a general cleanup policy or throughput benchmark.

## Further reading and attribution

- [State of Simulation for Physical AI](https://huggingface.co/blog/nvidia/state-of-simulation-for-physical-ai)
- [How to Use NVIDIA Warp and MuJoCo Warp](https://huggingface.co/blog/nvidia/how-to-use-nvidia-warp-and-mjwarp)
- [Newton source repository](https://github.com/newton-physics/newton)
- [NVIDIA learning path: Newton fundamentals](https://docs.nvidia.com/learning/physical-ai/getting-started-with-newton/latest/newton-fundamentals/overview.html)
- [Newton documentation](https://newton-physics.github.io/newton/stable/) and [solver coupling](https://newton-physics.github.io/newton/stable/concepts/coupling.html)
- [Third-party notices](notebooks/newton/THIRD_PARTY_NOTICES.md) and [retained licenses](notebooks/LICENSES/)

Interactive robot models download on first use from pinned upstream revisions. Immutable evidence archives also retain the exact historical source/model assets used to produce their records; see the package notices. The relocated companion code retains its MIT attribution to Johnny Nuñez Cano, including the frozen MuJoCo/MJWarp baselines.
