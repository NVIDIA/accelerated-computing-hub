# Article 2 — Warp and MJWarp

This Hub tutorial imports the source and evidence identified in [MIGRATION_MANIFEST.json](MIGRATION_MANIFEST.json).

Companion code for *[How to Use NVIDIA Warp and MJWarp to Accelerate Robotics Simulation and Learning Workflows](https://huggingface.co/blog/nvidia/how-to-use-nvidia-warp-and-mjwarp)*.

## Lessons and environment

| Notebook | Purpose |
|---|---|
| [MuJoCo fundamentals](part1/01__mujoco_fundamentals.ipynb) | Inspect and step a CPU scene |
| [Pick and place](part1/02__pick_and_place.ipynb) | Both robots place both cubes in the receiving box |
| [MuJoCo Warp](part2/03__mujoco_warp.ipynb) | The same task on one selected GPU |
| [CPU/GPU benchmark](part2/04__cpu_gpu_benchmark.ipynb) | Preflight by default; full task measurements are opt-in |

This update intentionally replaces the earlier Hub 3.12/1.17 package line with the measured Article 2 stack: **MuJoCo 3.8.0, MuJoCo Warp 3.8.0.3 and Warp 1.15.0**. Both local setup and the Brev image install the unchanged source hash lock. Notebook/UI tools are separate container additions. These imported workstation results are not new Hub, container, Brev or Colab measurements.

The [migration record](MIGRATION_MANIFEST.json) maps source files to this layout. [Archived evidence](benchmark-results/2026-10-09-validated-box-gpu0/README.md) retains its original paths, links and hashes. Use the current links here to run this relocated tutorial.

One pick-and-place task, two physics backends, two robots:

| Part | Backend | Notebooks |
|:----:|---------|-----------|
| **1** | `mujoco.mj_step` (CPU, float64) | `01__mujoco_fundamentals.ipynb`, `02__pick_and_place.ipynb` |
| **2** | `mujoco_warp.step` (GPU, float32) | `03__mujoco_warp.ipynb` |

Robots (switch with `--robot`):

- `so101` — [SO-101](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) (default) — model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Apache-2.0
- `rebot` — [Seeed reBot DevArm](https://github.com/google-deepmind/mujoco_menagerie/tree/main/seeed_rebot_devarm) — model from MuJoCo Menagerie, MIT (c) 2026 Seeed Studio

Robot models are downloaded on first use from pinned Menagerie commits.
The measured Blog 2 helper reuses an existing robot folder without rechecking its Git revision or tracked files. For a reproducible fresh download, choose an empty `MUJOCO_MENAGERIE_CACHE`; preserve existing/custom assets and record their identity separately.
Diagnostic archives also preserve the model assets used in those checks, with
their licenses. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for
licenses and attribution.

```bash
cd part1
python solutions/so101_pick_place_solution.py --task box --robot so101 --sim-substeps 20 --headless-steps 2000
python solutions/so101_pick_place_solution.py --task box --robot rebot --sim-substeps 20 --headless-steps 2000

cd ../part2
python solutions/so101_mjwarp_solution.py --task box --robot so101 --sim-substeps 20 --headless-steps 2000 --device cuda:0
python solutions/so101_mjwarp_solution.py --task box --robot rebot --sim-substeps 20 --headless-steps 2000 --device cuda:0
```

## Watch and stop the examples

To open the viewer, omit `--headless-steps` from a teaching command above. The viewer closes after the full episode: **40 simulated seconds for `--task box`**, or **12 for `--task stack`**. At the default 50 Hz, these are 2,000 and 600 control frames. Pausing extends the wall-clock duration without advancing the simulation.

The robot viewer uses public MuJoCo/GLFW APIs on the main thread. On macOS use ordinary environment `python`, not `mjpython`; macOS GUI execution has not been validated. The fundamentals sphere/passive-viewer instructions remain separate.

Press Space to pause or resume, Esc or Q to cancel, or Ctrl+C in the terminal to stop. Cancelling before the full episode skips the final task check. A completed box run performs the physical task checks. The stacking exercise prints `xy_err` and `dz` as diagnostics; it does not assert stacking success.

## Reproduce the verified environment

Use Python 3.12 and install from the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r tutorials/sim2real-blogs/notebooks/mujoco/requirements.lock.txt
source .venv/bin/activate
# Optional browser UI (the lock already supplies a notebook kernel):
uv pip install jupyterlab
jupyter lab
```

This Article 2 environment uses MuJoCo 3.8.0, MuJoCo Warp 3.8.0.3 and Warp
1.15.0, without Newton. It includes the notebook kernel and execution tools;
use that kernel in your notebook editor or install `jupyterlab` for a browser
interface. Article 3 has a separate Newton environment and its own benchmark protocol.

## Compare CPU and GPU throughput

The [CPU/GPU benchmark notebook](part2/04__cpu_gpu_benchmark.ipynb) and [measurement guide](BENCHMARK.md) configure the **same two-cube box task used in Article 3** at **1, 16, 32, 64, 128, 256, 512, 1024 and 2048 environments**, for both robots. Every environment must grasp, lift, carry and release both cubes inside the receiving box, then settle them and withdraw the gripper. Each episode lasts **40 simulated seconds**, with one warm-up and five measured repetitions. Native MuJoCo uses persistent CPU worker processes, up to 32; MuJoCo Warp batches the same commands on one GPU. Both record the same compact physical contact and containment evidence.

`migration_benchmark.py --task box` is the benchmark default. The [updated box-task results](benchmark-results/2026-10-09-validated-box-gpu0/README.md) cover both robots at all nine sizes on a shared AMD Ryzen Threadripper PRO 9985WX 64-Cores and one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`cuda:0`): **35 of 36 configurations passed; 1 excluded**. Every accepted configuration passed one complete warm-up and five measured episodes. See the [audited table and interpretation](BENCHMARK.md#measured-on-the-workstation) for matched CPU/GPU timings, including transfer and validation. Excluded configurations retain their evidence and supply no accepted timing or ratio.

The [earlier October 9 box study](benchmark-results/2026-10-09-box-sweep-rtx-pro-6000/README.md) remains available with its original configuration, timings and failures. It is separate historical evidence and is not paired with this updated sweep.

The [October 8 stacking study](benchmark-results/2026-10-08-full-sweep-rtx-pro-6000/README.md) remains historical stacking evidence, available through the explicit legacy `--task stack` protocol.

The notebook starts with setup and preflight only. Review memory, GPU selection and CPU allocation before enabling measurements. The guide includes isolated Colab setup and an existing Brev VM recipe. Article 2 keeps its own Python 3.12 dependency lock, separate from Newton in Article 3.

Run the notebook contract check with the Article 2 interpreter:

```bash
.venv/bin/python -m unittest discover -s tutorials/sim2real-blogs/test -p test_article2_benchmark_notebook.py
```

Set `ARTICLE2_NOTEBOOKS_EXECUTE=1` for an actual local Jupyter preflight run. This test keeps measurements disabled and leaves the published notebook unchanged.

## Inspect a single-world box task

The teaching notebooks run the completed `--task box` reference by default for both robots. Use the same explicit option from the terminal to run the shared benchmark task: pick up the red and blue cubes sequentially and release them inside a receiving box. CPU MuJoCo and single-world MJWarp share the scene, controller and measured checks in `box_task.py`. The box path uses the same solver settings and initial above-cube robot pose as the benchmark. The optional stack exercises remain available through `--task stack`; omitting `--task` in these legacy scripts still selects stacking.

From the module directory, run the completed references:

```bash
python part1/solutions/so101_pick_place_solution.py --task box --robot so101 --sim-substeps 20 --headless-steps 2000 --report part1/.generated/reference_box_so101.json
python part2/solutions/so101_mjwarp_solution.py --task box --robot so101 --sim-substeps 20 --headless-steps 2000 --device cuda:0 --report part2/.generated/reference_box_so101_cuda0.json
```

Use `--robot rebot` for the other arm, and `--device cuda:1` to select a second visible GPU. The 2,000-frame budget allows 40 simulated seconds at 50 Hz, with 20 physics substeps of 1 ms per frame; reaching the budget does not itself establish success. `--report` optionally saves a JSON summary when the run completes or exhausts its frame budget. An earlier simulation error exits nonzero before that report is written.

After completing the existing TODO exercises, select your student implementation directly:

```bash
python part1/so101_pick_place.py --task box --robot so101 --sim-substeps 20 --headless-steps 2000
python part2/so101_mjwarp.py --task box --robot so101 --sim-substeps 20 --headless-steps 2000 --device cuda:0
```

Reference runs call files under `solutions/`; they do not overwrite student files. The two task notebooks run the box reference first and expose an explicit reference/student selector. Earlier stack and microbenchmark exercises are optional. The teaching scripts run the box task in one world; their older `--benchmark --task box` combination is rejected. Use `migration_benchmark.py --task box` and the benchmark notebook for the full CPU/GPU box sweep.

See the [earlier verification record](VERIFICATION.md) for both robots, both GPU devices,
acceptance criteria and the exact environment. Newton is a separate tutorial and uses its own dependency environment.

## Current validation evidence

The [notebook execution record](notebook-validation/2026-10-10/README.md) covers the current box lessons and benchmark preflight. The [viewer record](VIEWER_VALIDATION.md) documents finite runs, pause, clean cancellation and the legacy solver warning correction.

## License and attribution

The imported tutorial retains [Johnny Nuñez Cano’s MIT license](LICENSE). Both original source notices are preserved in [LICENSES](LICENSES): [Johnny Nuñez Cano](LICENSES/Johnny-Nunez-Cano-MIT.txt) and [NVIDIA-dev](LICENSES/NVIDIA-dev-MIT.txt). Robot assets in retained diagnostic archives keep their upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
