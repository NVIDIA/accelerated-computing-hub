# Newton Robot Tasks

Build a small Newton simulation, migrate a MuJoCo task that picks up both cubes and places them in a receiving box, and then use a robot gripper to place two cubes, a cloth patch and a cable into a box. The coupled task assigns the robot and cubes to MuJoCo, and the cloth and cable to Vertex Block Descent (VBD). Newton manages the shared scene, state, controls and contact interfaces. On CUDA, MuJoCo Warp advances the rigid solver.

This tutorial is self-contained. It includes the earlier MuJoCo/MJWarp **Python baselines** for comparison, so no other Hub tutorial or pull request is required. SO-101 and Seeed reBot DevArm share the learning path.

## Start here

| Lesson | What you learn |
|---|---|
| [Newton fundamentals](notebooks/newton/part3/01__newton_fundamentals.ipynb) | Build, collide and step a scene; choose a solver; replicate worlds |
| [MuJoCo to Newton](notebooks/newton/part3/02__mujoco_to_newton.ipynb) | Preserve the two-cube task while mapping joints, controls, states and contacts |
| [Clean the table](notebooks/newton/part3/03__clean_the_table.ipynb) | Author the box, free cloth and rod; couple solvers; grasp, carry and release all four payloads |
| [Final check](notebooks/newton/part3/04__final_check.ipynb) | Run both CPU box tasks; optionally check CUDA, stacking and solver coupling |
| [CPU/GPU benchmark](notebooks/newton/part3/05__migration_benchmark.ipynb) | Inspect preflight, then compare matched batches from 1 to 2048 worlds |

The four teaching notebooks default to `REFERENCE = True`, selecting the complete external Python solutions. Set it to `False` after completing the `TODO Step` sections in the starter scripts. Reference runs never copy a solution over your exercise files, and backups preserve existing work. There are no separate solution notebooks: each notebook explicitly chooses the reference or student Python implementation.

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

Notebooks 02 and 04 run the full 40-second box task on CPU for both robots; CUDA is opt-in. Notebook 05 defaults to `RUN_BENCHMARK = False` and runs preflight only. Notebook 03 is the separate, slower coupled-material lesson: historical CPU rollouts took approximately 26–29 minutes per robot on the recorded workstation. Long cells capture subprocess output until completion. See the [setup and task guide](notebooks/newton/README.md) for device selection, cache remedies, rendering and student/reference commands.

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
