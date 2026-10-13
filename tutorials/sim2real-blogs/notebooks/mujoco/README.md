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

**Setup:** follow the [Quickstart](../../README.md#quickstart-local) (one environment,
one install command, then `python doctor.py`). This guide covers the lessons.

Robot models are downloaded automatically on first use from pinned Menagerie
commits; no environment variable is needed. The automatic cache is versioned by the
pinned commit (`~/.cache/mujoco_menagerie/<commit>/`), reused only when Git confirms
the pinned revision and unchanged robot files, and downloaded into a temporary folder
that is renamed into place after that check. Anything else is moved aside, never
deleted. Explicit `--menagerie-path` / `MUJOCO_MENAGERIE_PATH` folders are used as
given; a path without the robot folder prints one note and the download continues.
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

To open the viewer, omit `--headless-steps` from a teaching command above. The viewer closes by itself after the full episode: **40 simulated seconds for `--task box`**, or **12 for `--task stack`** (the default). At 50 Hz these are 2,000 and 600 control frames. When rendering is slower than real time (software OpenGL, small laptop GPUs), the loop simulates several frames per rendered frame instead of slowing the physics down; physics frames are never skipped.

| Input | Effect |
| --- | --- |
| Space | pause / resume physics and controller (prints the frame it paused and resumed at) |
| Esc or Q, or closing the window | stop now; the task check is skipped |
| Ctrl+C in the terminal | stop within about a second in every phase (setup, kernel compilation, viewer, check); a second Ctrl+C exits immediately |

Every run ends with one explicit line and exit code:

| Last line | Exit code |
| --- | --- |
| `Task complete: red cube stacked on blue (verified: xy_err=…, dz=…)` or `Task complete: both cubes are in the receiving box (verified).` | 0 |
| `Task FAILED: …` with the measured values | 1 |
| `Stopped by Esc / Q / window close after N/600 frames … the task check was skipped` | 0 |
| `Stopped by Ctrl+C …` or `Interrupted by Ctrl+C. Exiting.` | 130 |

The stack check uses the article's thresholds: horizontal centre error ≤ 15 mm, vertical centre separation 35–55 mm, blue cube still on the table and both cubes at rest. The box check runs the full grasp, lift, carry, release and settle evidence from `box_task.py`.

The window is a small synchronous GLFW viewer owned by the main thread (`viewer_window.py`, `viewer_loop.py`), not `mujoco.viewer.launch_passive`. Earlier copies of these lessons, including the previous version of this tutorial, used `launch_passive`: that window never closed after the task, ignored Esc, and Ctrl+C left a traceback and could hang on `GLXBadDrawable` on NVIDIA drivers. On macOS, start these scripts with ordinary environment `python`, not `mjpython`; macOS GUI execution has not been validated. The fundamentals sphere/passive-viewer instructions remain separate.

## Run the blog's Part 2 commands

The article's Part 2 block runs two different programs. The first is the one-world correctness check: it shows the same pick-and-place as Part 1 (now stepped by `mujoco_warp.step`), so it looks like Part 1 by design, and it closes after 12 simulated seconds. Throughput comes only from `--benchmark` and `scaling_study.py`:

```bash
cd part2
python solutions/so101_mjwarp_solution.py                       # Gate 2: one world on the GPU, viewer, verified stack
python solutions/so101_mjwarp_solution.py --benchmark --nworld 2048 --steps 200   # one batched measurement
python scaling_study.py --worlds 1 64 1024 2048 8192 --steps 100                   # the scaling curve
```

## Reading the benchmark numbers

`--benchmark` and `scaling_study.py` print a header that states what is measured, the GPU (SM count, VRAM), the CPU, the physics timestep and a single-core classic-MuJoCo baseline, then one row per batch size and a short reading guide. All columns are aggregate over the batch:

| Column | Meaning |
| --- | --- |
| `nworld` | independent copies of the scene advanced together by one `mjw.step` |
| `ms/step` | wall-clock time of one batched step (latency of the whole batch) |
| `world-steps/s` | `nworld × steps / seconds`: physics samples per second, the number that matters for RL and data generation |
| `vs 1 world` | throughput relative to `nworld = 1` on the same GPU |
| `vs CPU core` | throughput relative to **one** CPU core running classic `mujoco.mj_step` on the same scene |
| `sim-s/s` | simulated seconds per wall-clock second over all worlds (aggregate real-time factor) |
| `peak VRAM` | Warp memory-pool high-water mark for that batch, temporaries included |

This is a physics-step microbenchmark: the arm holds its start pose with fixed controls, no IK, no viewer, one CUDA graph replayed `--steps` times after 10 warm-up steps. It shows how a GPU scales; it is not the full task. The [box-task study](BENCHMARK.md) measures the complete two-cube task against 32 CPU worker processes.

**Measured** on this repository's workstation (one NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 188 SMs, 96 GB; AMD Ryzen Threadripper PRO 9985WX), `python scaling_study.py --robot <robot> --worlds 1 16 64 256 1024 2048 4096 8192 16384 32768 --steps 200`, 2 ms timestep, 13 October 2026:

| nworld | SO-101 ms/step | SO-101 world-steps/s | SO-101 peak VRAM | reBot ms/step | reBot world-steps/s | reBot peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.373 | 2,678 | 10 MiB | 0.327 | 3,059 | 15 MiB |
| 64 | 0.484 | 132,260 | 61 MiB | 0.388 | 164,986 | 303 MiB |
| 1024 | 0.649 | 1,578,877 | 835 MiB | 0.514 | 1,990,906 | 4.6 GiB |
| 2048 | 0.748 | 2,739,444 | 1.6 GiB | 0.609 | 3,364,472 | 9.2 GiB |
| 4096 | 1.024 | 3,999,088 | 3.2 GiB | 0.867 | 4,725,259 | 18.3 GiB |
| 8192 | 1.664 | 4,924,047 | 6.5 GiB | 1.498 | 5,468,899 | 36.6 GiB |
| 16384 | 3.423 | 4,786,501 | 12.9 GiB | — | out of memory | — |
| 32768 | 7.622 | 4,299,087 | 25.8 GiB | — | — | — |

One CPU core ran 46,000–60,000 `mj_step`/s on these scenes (about 100× real time). How to read it:

- Up to about 1,024–2,048 worlds, `ms/step` barely moves while `nworld` grows 1,000×: the GPU still has idle SMs, so throughput rises almost linearly. Past about 4,096–8,192 worlds `ms/step` grows in proportion to `nworld` and throughput plateaus (≈ 4–5 million world-steps/s here). The knee is the useful batch size for this GPU and scene. Run-to-run variation on the plateau is about ±10 %.
- One world on the GPU (≈ 2,700 steps/s) is about 20× slower than one CPU core: MJWarp is for batches, not single-world latency.
- `vs CPU core` is one core. For a fair comparison with a CPU farm, divide by the cores you would use; on the full task the GPU overtakes 32 CPU workers from about 1,024 worlds ([BENCHMARK.md](BENCHMARK.md#measured-on-the-workstation)).
- VRAM caps `nworld`. The reBot scene needs about 4.6 MiB per world at the peak (contact and constraint buffers sized by `nconmax`/`njmax`, plus collision temporaries), so 16,384 reBot worlds did not fit in 96 GB; SO-101 needs about 0.8 MiB per world. An `out of memory` message ends the sweep cleanly; lower `nworld` or tighten `nconmax`/`njmax`.

**Expected, not measured**, on other GPUs (plateau throughput scales roughly with SM count × clock and memory bandwidth; the batch limit with VRAM; small-batch latency is mostly kernel-launch and driver overhead and changes little):

| GPU | SMs / bandwidth / VRAM | Expected plateau vs the RTX PRO 6000 numbers above | Expected largest batch (from the measured peak VRAM) |
| --- | --- | --- | --- |
| RTX 5090 | 170 / 1.8 TB/s / 32 GB | about 85–100 % (same Blackwell generation, slightly fewer SMs) | SO-101 ≈ 16,384–32,768 (32,768 needed 25.8 GiB here); reBot 4,096 (8,192 needs ~37 GiB and should run out of memory) |
| RTX 3060 (12 GB) | 28 / 360 GB/s / 12 GB | about 10–20 %, with the knee at roughly 1,024–2,048 worlds | SO-101 8,192 (6.5 GiB); reBot 1,024–2,048 (2,048 needed 9.2 GiB) |

Send measured results from those GPUs with `nvidia-smi` output; they replace these expectations.

## Reproduce the verified environment

The [Quickstart](../../README.md#quickstart-local) creates it: Python 3.12 with
MuJoCo 3.8.0, MuJoCo Warp 3.8.0.3 and Warp 1.15.0 from the unchanged hash lock,
without Newton, then `python doctor.py` checks it. The lock includes the notebook
kernel and execution tools; install `jupyterlab` for a browser interface.
Article 3 has a separate Newton environment and its own benchmark protocol.

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

The [13 October viewer and benchmark record](validation/viewer-exit-and-benchmark-2026-10-13/README.md) reproduces the tester's report and validates the fix. The [notebook execution record](notebook-validation/2026-10-10/README.md) covers the current box lessons and benchmark preflight. The [viewer record](VIEWER_VALIDATION.md) documents finite runs, pause, clean cancellation and the legacy solver warning correction.

## License and attribution

The imported tutorial retains [Johnny Nuñez Cano’s MIT license](LICENSE). Both original source notices are preserved in [LICENSES](LICENSES): [Johnny Nuñez Cano](LICENSES/Johnny-Nunez-Cano-MIT.txt) and [NVIDIA-dev](LICENSES/NVIDIA-dev-MIT.txt). Robot assets in retained diagnostic archives keep their upstream licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
