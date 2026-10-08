# Newton Robot Tasks

Build a small Newton simulation, migrate a MuJoCo two-cube stack, and then use a robot gripper to place two cubes, a cloth patch and a cable into a box. The coupled task assigns the robot and cubes to MuJoCo, and the cloth and cable to Vertex Block Descent (VBD). Newton manages the shared scene, state, controls and contact interfaces. On CUDA, MuJoCo Warp advances the rigid solver.

This tutorial is self-contained. It includes the earlier MuJoCo/MJWarp **Python baselines** for comparison, so no other Hub tutorial or pull request is required. SO-101 and Seeed reBot DevArm share the learning path.

## Start here

| Lesson | What you learn |
|---|---|
| [Newton fundamentals](notebooks/newton/part3/01__newton_fundamentals.ipynb) | Build, collide and step a scene; choose a solver; replicate worlds |
| [MuJoCo to Newton](notebooks/newton/part3/02__mujoco_to_newton.ipynb) | Preserve the two-cube task while mapping joints, controls, states and contacts |
| [Clean the table](notebooks/newton/part3/03__clean_the_table.ipynb) | Author the box, free cloth and rod; couple solvers; grasp, carry and release all four payloads |
| [Final check](notebooks/newton/part3/04__final_check.ipynb) | Check both robots and inspect measured grasp-to-release outcomes on the selected device |
| [When is migration worth it?](notebooks/newton/part3/05__migration_benchmark.ipynb) | Measure native CPU MuJoCo and the GPU workflow with recorded hardware, configuration and task outcomes |

The four teaching notebooks default to `REFERENCE = True`, selecting the complete external Python solutions. Set it to `False` after completing the `TODO Step` sections in the starter scripts. Reference runs never copy a solution over your exercise files, and backups preserve existing work. There are no separate solution notebooks: each lesson explicitly chooses the reference or student Python implementation. The optional benchmark notebook runs preflight first and collects measurements only when enabled.

## Run locally

Use **Python 3.12**. From this directory:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r notebooks/newton/requirements.lock.txt
source .venv/bin/activate
PXR_WORK_THREAD_LIMIT=1 jupyter lab --notebook-dir notebooks
```

Open the lessons inside `newton/part3/` and use that directory as the kernel's working directory. The first cell prints the actual interpreter and checks Newton 1.6.0. Notebook metadata follows the Hub's common template; its Python/T4 metadata is not the tutorial's dependency specification or a Colab validation claim.

The pinned stack uses Newton 1.6.0, Warp 1.17.0, MuJoCo/MuJoCo Warp 3.12.0, OpenUSD 26.3, NumPy 2.5.3 and SciPy 1.18.1. Linux x86-64 requires glibc 2.35 or newer for the included Open3D wheel. CPU execution also works on macOS ARM64; CUDA requires a supported NVIDIA GPU. Launch with `PXR_WORK_THREAD_LIMIT=1` **before** importing OpenUSD and restart existing kernels after changing it.

The first two lessons are short after downloads and compilation. The full CPU path repeats three coupled rollouts: on the verified workstation each took approximately 26–29 minutes, with all four notebooks taking about 83 minutes. Long cells capture subprocess output until completion. See the [setup and task guide](notebooks/newton/README.md) for device selection, cache remedies, rendering and student/reference commands.

## Docker and Brev

The [Compose configuration](brev/docker-compose.yml) supplies the repository's standard `base` and `jupyter` services. It installs the same hash-locked Python 3.12 environment and sets the OpenUSD launch variable. The CUDA 13.1 image needs a compatible host driver and NVIDIA Container Toolkit; the services expose JupyterLab on port 8888.

From the Hub repository root:

```bash
brev/dev-build.bash newton
brev/dev-start.bash newton
brev/dev-test.bash newton
```

The default test entrypoint runs the regression suite, the first two teaching notebooks and the benchmark preflight. To run all four teaching notebooks, set `NEWTON_NOTEBOOKS_INCLUDE_FINAL=1` before starting Compose, or run `NEWTON_NOTEBOOKS_INCLUDE_FINAL=1 bash brev/test.bash` inside this tutorial's environment. `brev/test.bash 03` or `04` explicitly selects a long coupled-task notebook; standard pytest paths and flags are also accepted.

The Docker image built successfully on the verification workstation, and its default suite passed 119 tests, including the first two notebooks; the two long notebook tests were explicitly skipped. A live Brev deployment and Colab have not been validated. The original full physics validation and the relocation/container checks are distinguished in [VERIFICATION.md](notebooks/newton/VERIFICATION.md).

## Measure migration on your hardware

The [benchmark notebook](notebooks/newton/part3/05__migration_benchmark.ipynb) starts with one world and 16 worlds. It runs preflight first; measurement is explicit. Its replay experiment uses the same MJCF model and task commands on native MuJoCo and MuJoCo Warp, comparing CPU serial, CPU worker-pool and GPU batch timings. An optional one-world workflow experiment includes the actual host controller and Newton migration, with its model/contact differences reported separately.

The reports save configuration, hardware/software metadata, repeated samples and task checks as JSON, CSV and Markdown. These are community measurements on the specified machine, not official product benchmarks. The [benchmark guide](notebooks/newton/BENCHMARK.md) explains how to interpret the results and run the same code locally, on an existing Brev instance, or through the notebook's isolated Colab setup. Lower-end GPU and hosted-runtime validation remain explicit follow-up work; included launch code does not establish a tested deployment.

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

Robot models download on first use from pinned upstream revisions; the mesh files themselves are not vendored. The relocated companion code retains its MIT attribution to Johnny Nuñez Cano, including the frozen MuJoCo/MJWarp baselines.
