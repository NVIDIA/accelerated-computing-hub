# Newton CPU and GPU throughput for the two-cube box task

Use the [benchmark notebook](part3/05__migration_benchmark.ipynb) or `part3/migration_benchmark.py` to measure actual Newton execution on CPU and one GPU. Each environment must **pick up both cubes, carry them to the receiving box, release them inside and withdraw the gripper**. The full sweep covers **1, 16, 32, 64, 128, 256, 512, 1024 and 2048 environments** for both `so101` and `rebot`, with **one complete warm-up and five measured repetitions** per configuration.

The [updated box-task results](benchmark-results/2026-10-09-validated-box-gpu0/README.md) cover both robots at all nine sizes on a shared AMD Ryzen Threadripper PRO 9985WX 64-Cores and one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`cuda:0`): **35 of 36 configurations passed; 1 excluded**. Every accepted configuration passed one complete warm-up and five measured episodes. See the [audited table and interpretation](BENCHMARK.md#measured-on-the-workstation) for matched CPU/GPU timings, including transfer and validation. Excluded configurations retain their evidence and supply no accepted timing or ratio.

The [earlier October 9 box study](benchmark-results/2026-10-09-box-sweep-rtx-pro-6000/README.md) remains available with its original configuration, timings and failures. It is separate historical evidence and is not paired with this updated sweep.

The [earlier October 8 workstation study](benchmark-results/2026-10-08-blogs-rtx-pro-6000/README.md) remains historical stacking data, with different replay and single-world workflow measurements.

Article 3 keeps its Python 3.12 hash lock, including **Newton 1.6.0, Warp 1.17.0 and MuJoCo/MuJoCo Warp 3.12.0**. Article 2 uses the same box-task duration, observation format and acceptance rules with its own pinned native MuJoCo/MuJoCo Warp stack.

## Measured on the workstation

We compared Newton SolverMuJoCo on CPU and CUDA on the same task with SO-101 and reBot: pick up both cubes, place them in the receiving box, and withdraw the gripper. Each world completes 40 simulated seconds. We tested nine batch sizes, from one to 2048 independent worlds.

The table shows median simulation seconds across five repetitions after one full warm-up, measured on shared AMD Ryzen Threadripper PRO 9985WX 64-Cores and one NVIDIA RTX PRO 6000 Blackwell Workstation Edition. The CPU uses up to 32 persistent workers, capped by world count; the GPU uses one device. Reset, commands, integration and grasp-verification observations are included. Inverse kinematics is precomputed before timing and replayed identically on both backends.

| Worlds | SO-101 CPU (s) | SO-101 GPU (s) | reBot CPU (s) | reBot GPU (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.4848 | 14.7928 | 6.3294 | 15.5382 |
| 16 | 7.2303 | 16.7782 | 7.4193 | 17.1097 |
| 32 | 7.6543 | 17.2842 | 7.4869 | 17.6607 |
| 64 | 15.2200 | 17.8851 | 15.1986 | 18.6563 |
| 128 | 30.4011 | 18.7815 | 29.9407 | 19.3498 |
| 256 | 60.7351 | 19.9407 | 59.7683 | 20.6777 |
| 512 | 122.6874 | 21.6250 | 119.8159 | 22.3135 |
| 1024 | 242.4549 | 23.8462 | 238.9082 | Excluded |
| 2048 | 482.4144 | 28.3054 | 475.6768 | 30.0213 |

For SO-101, the first tested batch faster on GPU was 128 worlds; the CPU took 17.04 times as long at 2048 worlds. For reBot, the first tested batch faster on GPU was 128 worlds; the CPU took 15.84 times as long at 2048 worlds. These observations apply to the tested sizes, rather than universal crossover thresholds.

Including host output collection and task validation: For SO-101, the first tested batch faster on GPU was 128 worlds; the CPU took 14.00 times as long at 2048 worlds. For reBot, the first tested batch faster on GPU was 128 worlds; the CPU took 13.19 times as long at 2048 worlds. These medians sum all three costs within each repetition. Setup, compilation, warm-up and validator teardown remain separate.

Excluded configurations: reBot GPU at 1024 worlds. 35 of 36 configurations passed every required episode. Every accepted world must physically grasp, lift, carry, release and settle both cubes inside the box. One failed episode excludes the entire configuration and its timing from comparisons; exact physical and numerical checks are in the guide.

Use the [benchmark notebook](part3/05__migration_benchmark.ipynb) to reproduce the task. The [companion benchmark guide](BENCHMARK.md) explains the measurement method; [full results](benchmark-results/2026-10-09-validated-box-gpu0/README.md) retain ratios, ranges, failures and hardware configuration. These community measurements are not official product benchmarks or evidence of lower-end GPU, Colab or Brev performance.

The companion also includes an [ALOHA pot-and-lid replay](reference-benchmark/README.md) comparing native MuJoCo on CPU with MuJoCo Warp on GPU. It is a separate workload and does not measure Newton's API or this box task.

## Choose the comparison

| Scope | Task and execution |
|---|---|
| `newton-batch` (CLI and notebook default) | Two cubes into the receiving box. Newton `SolverMuJoCo` runs in persistent CPU worker processes or one replicated CUDA model, with the same recorded position-target tape. |
| `replay` | Historical 12-second stacking task. Native MuJoCo serial/pool execution and MuJoCo Warp replay the same compiled MJCF and controls. |
| `workflow` | Historical single-world stacking task with the online host controller, through native MuJoCo and Newton CPU/CUDA. |
| `both` | Runs the legacy `replay` and `workflow` experiments separately; it does not include the box task. |

The box benchmark uses commands already available. Online inverse kinematics, rendering, policy training and the coupled cloth-and-cable exercise are outside its timer. Legacy stacking results retain their original task, recording policy and timing scope; they cannot substitute for a box-task result.

## Newton CPU vs Newton GPU

Both backends use a 330 × 250 × 55 mm receiving-box interior and separate placement targets. The gripper lowers each held cube before opening, with a nominal 32 mm gap above the floor. The receiving floor uses direct `solref = (-4444.444444, -266.666667)`, retaining the mixed normal stiffness and doubling normal damping. Other materials, friction coefficients and default collision routes remain unchanged. The original physical grasp, lift, carry, release and settling gates still apply.

`--scope newton-batch` selects the canonical `BoxTask` sequence on both devices. At reset, the arm starts at the first approach target above the red cube to avoid intersecting the table or box; both cubes remain at their original outside-box spawn positions. Each environment consumes the same precomputed float32 target tape for **2,000 control frames at 50 Hz**, with **20 physics steps per frame at 0.001 s**: **40,000 steps and 40 simulated seconds**. Keep this horizon fixed at every batch size, including the final settling and withdrawal period. Both devices use MuJoCo contact detection, 100 solver iterations, 50 line-search iterations and `impratio = 100`. Reports preserve the native model settings and the actual uploaded GPU settings separately. Both backends request solver tolerance `1e-6` and line-search tolerance `0.01`. Native CPU integration uses float64 and CUDA uses float32. The benchmark retains each backend’s default collision route. Newton’s default `enable_multiccd=False` is retained on both devices and appears as `disableflags = 524288` in these reports.

The CPU baseline uses persistent worker **processes**, each with its own single-world Newton model and `SolverMuJoCo`. `--cpu-threads 32` requests up to 32 workers, capped by environment count. CUDA advances one replicated Newton model on the selected GPU. The CPU baseline actually runs Newton; direct native MuJoCo rollout is confined to the legacy replay scope.

This single-`SolverMuJoCo` task uses `update_data_interval=0` on both devices: reset synchronizes the initial state, MuJoCo retains its evolving coordinates, and commands, forces and Newton output state still update each step. This avoids repeated coordinate round trips; coupled solvers that modify each other’s state retain their separate policy.

The simulation timer includes episode reset, target commands, Newton physics stepping, per-frame forward evaluation for current geometry and solved contact forces, and observation recording. CPU dispatch/completion and CUDA synchronization are included. Command generation, model/solver setup, compilation, warm-up, final host collection/download and validation are separate costs. Reports retain simulation time and time to checked host results.

Both backends use the same requested host-validation budget (`--cpu-threads 32`), capped by world count and CPU affinity. One world validates in the parent; larger batches reuse CPU workers and read-only shared histories with CUDA hidden. Validation time includes history copy, dispatch, physical/clock checks and collection; worker startup and teardown are recorded separately.

Both devices record **61 float32 fields per environment at 50 Hz**: time, gripper opening and final-frame tool clearance (zero in earlier frames); then, for each cube, eight 3D corners, maximum corner speed, two jaw-contact counts and two solved jaw normal forces. The complete 2,048-environment history occupies **999,424,000 bytes (953.125 MiB)** before host copies and additional model, solver or scratch allocations.

Newton's per-device conversion can produce different compiled-model hashes. Reports retain these hashes and the shared protocol signature. Compare compatible, successful CPU/CUDA rows at the same environment count; CPU time divided by CUDA time is below 1 when CUDA takes longer. This measures the Newton application's batch cost.

## What counts as a successful task

The same `CubeEvidence` rules apply to both cubes, in every world and every warm-up or measured repetition:

1. Grasp with contact on both jaws and solved normal force above **1e-5 N per jaw** for at least **10 consecutive frames**, establishing the grasp during closing.
2. Keep the whole cube at least **1 cm above the table** for **25 consecutive loaded frames**.
3. Retain the loaded grasp during at least **10 carry frames** and at least **5 cm of uninterrupted airborne XY travel**; all eight corners must reach over the box before opening.
4. Open the actual gripper to at least **80%**, then detach the cube from both jaws. Dropping it before release fails the task.
5. Keep all eight cube corners inside the receiving box, within **3 mm** tolerance, with maximum corner speed below **0.04 m/s** for **50 consecutive detached frames**.

The red cube must be released before the blue cube is grasped. At completion, both cubes must remain contained and settled, and the gripper must withdraw at least **2 cm above the higher of the table and box rim**. Finite observations, complete clock progression, all 40,000 actual physics steps and valid capacity checks are also required. The raw clock is checked against accumulation at the backend’s own precision; the float32 GPU clock can differ slightly from nominal elapsed time. An independent integration counter prevents a missing or duplicated step from passing that check. An object center inside the box, a phase label or a successful process exit is not sufficient evidence.

Capacity overflow and invalid time/state observations fail the configuration. Solver iteration-limit diagnostics remain visible. Every warm-up and all five measured repetitions must pass for a timing or ratio to be accepted. The first failed episode stops that configuration; the report retains the completed episodes and the sweep continues with the other configurations. Preserve failures and missing repetitions; do not plot partial-run times or convert them to zero.

## Run on your machine

Prepare the [Article 3 environment](README.md#setup), then run from `tutorials/newton/notebooks/newton/part3`:

```bash
python migration_benchmark.py --scope newton-batch --preset workstation --device cuda:0 --cpu-threads 32 --nconmax 64 --njmax 128 --timeout 7200 --preflight
python migration_benchmark.py --scope newton-batch --robot so101 \
  --worlds 1 16 32 64 128 256 512 1024 2048 --cpu-threads 32 --device cuda:0 \
  --nconmax 64 --njmax 128 --repeats 5 --warmups 1 --order-seed 0 \
  --timeout 7200 --max-trajectory-mib 8192 --output-dir .generated/box-full-so101
```

Requested box capacities are `nconmax=64` and `njmax=128`. Reports retain actual allocated limits and high-water counts; capacity overflow invalidates a case. The 7,200-second case timeout does not shorten the 40-second simulated episode.

Repeat with `--robot rebot` and a fresh directory. The Newton batch scope fixes the task to `box`; no separate `--task` option is required. `--preset workstation` contains all nine sizes; `--preset developer` selects 1 and 16. Explicit `--worlds` overrides the preset. A smaller list is an exploratory subset, not a full sweep.

Preflight records devices, configuration and observation-memory estimates without producing performance numbers. The 8 GiB history budget does not cover every model, solver or scratch allocation and does not guarantee a row will fit. Review CPU allocation, available RAM/VRAM and concurrent applications. An oversized configuration fails instead of shortening the episode.

`--cpu-only` leaves GPU performance unmeasured. `--device cuda:0` selects the first visible CUDA device; choose the available device on your machine. Each invocation uses one GPU. The notebook defaults to `RUN_BENCHMARK = False`, so Run All performs setup and preflight only. Review its printed command before enabling measurements.

## Save and interpret reports

Each output directory contains `results.json`, `summary.csv`, `summary.md` and raw worker reports/logs. Preserve the source revision, package lock, CPU/GPU/RAM, driver, environment counts, worker allocation, individual samples and task outcomes. Use a new directory for each run; compiled caches are not measurement evidence.

Report CPU and GPU medians and observed minimum–maximum across the five accepted repetitions, with the actual CPU worker count. Include setup/compilation costs when they matter to a short-lived application. These community measurements describe the recorded machine and its load; they do not establish performance on another workstation or cloud GPU.

One environment tests latency; a large batch tests throughput. GPU execution can take longer at small batch sizes even when a larger batch is faster. [MuJoCo’s MJWarp guidance](https://mujoco.readthedocs.io/en/latest/mjwarp/#low-latency) describes this distinction. The sweep measures whether and where a crossover occurs for this task; GPU acceleration is not a condition for accepting a valid result.

## Separate ALOHA reference benchmark

The [ALOHA reference benchmark](reference-benchmark/README.md) compares native MuJoCo with MuJoCo Warp on a separate reference workload. It measures neither Newton’s API path nor this two-cube receiving-box task. Keep its timings and hardware/configuration records separate from the box results.

## Google Colab

Open [the benchmark notebook on Colab](https://colab.research.google.com/github/johnnynunez/accelerated-computing-hub/blob/2d100e859b174ab67834d09c0f705d09a5b0d516/tutorials/newton/notebooks/newton/part3/05__migration_benchmark.ipynb) and select a GPU runtime. The setup checks out `REPO_URL` at `REF`, installs uv 0.12.5 into a tools environment, and syncs the hash lock into a separate Python 3.12 environment. Benchmark subprocesses use that interpreter; the hosted kernel remains separate. `REF` selects the public Hub source revision; retain the printed resolved commit with any new results. The public fork branch hosts this update before merge. The notebook records the fetched source revision; Colab and lower-end GPUs remain unvalidated.

Review preflight on the assigned CPU/GPU before enabling measurements. Hosted hardware and runtime limits vary; this launch recipe is not a validated Colab box-task result. Download the report archive before disconnection. The CPU baseline belongs to the runtime.

## NVIDIA Brev

An existing Brev GPU VM can use the same repository and lock:

```bash
git clone --branch feature/blog3-validated-box-benchmark https://github.com/johnnynunez/accelerated-computing-hub.git
cd accelerated-computing-hub/tutorials/newton
git rev-parse HEAD  # Retain this resolved Hub revision with your results.
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes \
  notebooks/newton/requirements.lock.txt
PXR_WORK_THREAD_LIMIT=1 .venv/bin/python notebooks/newton/part3/migration_benchmark.py \
  --scope newton-batch --preset workstation --nconmax 64 --njmax 128 --timeout 7200 --preflight
```

Use that interpreter for the measurement command or notebook. Review the instance allocation and preserve reports before stopping the VM. Box-task validation on Brev is pending.
