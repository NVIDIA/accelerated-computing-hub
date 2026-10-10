# CPU and GPU throughput for the two-cube box task

Use the [benchmark notebook](part2/04__cpu_gpu_benchmark.ipynb) or `part2/migration_benchmark.py` to compare native MuJoCo on CPU with MuJoCo Warp on one GPU. Each environment must **pick up both cubes, carry them to the receiving box, release them inside and withdraw the gripper**. Run **1, 16, 32, 64, 128, 256, 512, 1024 and 2048 environments** for both `so101` and `rebot`, with **one complete warm-up and five measured repetitions** per configuration.

The [updated box-task results](benchmark-results/2026-10-09-validated-box-gpu0/README.md) cover both robots at all nine sizes on a shared AMD Ryzen Threadripper PRO 9985WX 64-Cores and one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`cuda:0`): **35 of 36 configurations passed; 1 excluded**. Every accepted configuration passed one complete warm-up and five measured episodes. See the [audited table and interpretation](BENCHMARK.md#measured-on-the-workstation) for matched CPU/GPU timings, including transfer and validation. Excluded configurations retain their evidence and supply no accepted timing or ratio.

The [earlier October 9 box study](benchmark-results/2026-10-09-box-sweep-rtx-pro-6000/README.md) remains available with its original configuration, timings and failures. It is separate historical evidence and is not paired with this updated sweep.

The [October 8 stacking study](benchmark-results/2026-10-08-full-sweep-rtx-pro-6000/README.md) remains historical; its raw reports and figures describe a different task.

Article 2 keeps its Python 3.12 hash lock: **MuJoCo 3.8.0, MuJoCo Warp 3.8.0.3 and Warp 1.15.0**, without Newton. Article 3 uses the same box-task duration, observation format and acceptance rules in its separate Newton environment.

## Measured on the workstation

We compared native MuJoCo CPU and MuJoCo Warp on the same task with SO-101 and reBot: pick up both cubes, place them in the receiving box, and withdraw the gripper. Each world completes 40 simulated seconds. We tested nine batch sizes, from one to 2048 independent worlds.

The table shows median simulation seconds across five repetitions after one full warm-up, measured on shared AMD Ryzen Threadripper PRO 9985WX 64-Cores and one NVIDIA RTX PRO 6000 Blackwell Workstation Edition. The CPU uses up to 32 persistent workers, capped by world count; the GPU uses one device. Reset, commands, integration and grasp-verification observations are included. Inverse kinematics is precomputed before timing and replayed identically on both backends.

| Worlds | SO-101 CPU (s) | SO-101 GPU (s) | reBot CPU (s) | reBot GPU (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2682 | 15.6355 | 1.1887 | 19.4162 |
| 16 | 1.4631 | 17.9851 | 1.3369 | 21.5820 |
| 32 | 1.4754 | 19.3022 | 1.3536 | 23.0548 |
| 64 | 2.9472 | 20.5499 | 2.6919 | 24.0782 |
| 128 | 5.8159 | 22.6108 | 5.3933 | 25.4998 |
| 256 | 11.7220 | 24.0796 | 10.6702 | 27.3368 |
| 512 | 23.2236 | 25.7572 | 21.1583 | Excluded |
| 1024 | 45.8203 | 29.2140 | 42.0190 | 32.0479 |
| 2048 | 90.2232 | 34.4835 | 83.1366 | 38.1011 |

For SO-101, the first tested batch faster on GPU was 1024 worlds; the CPU took 2.62 times as long at 2048 worlds. For reBot, the first tested batch faster on GPU was 1024 worlds; the CPU took 2.18 times as long at 2048 worlds. These observations apply to the tested sizes, rather than universal crossover thresholds.

Including host output collection and task validation: For SO-101, the first tested batch faster on GPU was 1024 worlds; the CPU took 2.36 times as long at 2048 worlds. For reBot, the first tested batch faster on GPU was 1024 worlds; the CPU took 2.00 times as long at 2048 worlds. These medians sum all three costs within each repetition. Setup, compilation, warm-up and validator teardown remain separate.

Excluded configurations: reBot GPU at 512 worlds. 35 of 36 configurations passed every required episode. Every accepted world must physically grasp, lift, carry, release and settle both cubes inside the box. One failed episode excludes the entire configuration and its timing from comparisons; exact physical and numerical checks are in the guide.

Use the [benchmark notebook](part2/04__cpu_gpu_benchmark.ipynb) to reproduce the task. The [companion benchmark guide](BENCHMARK.md) explains the measurement method; [full results](benchmark-results/2026-10-09-validated-box-gpu0/README.md) retain ratios, ranges, failures and hardware configuration. These community measurements are not official product benchmarks or evidence of lower-end GPU, Colab or Brev performance.

The companion also includes an [ALOHA pot-and-lid replay](reference-benchmark/README.md) comparing native MuJoCo on CPU with MuJoCo Warp on GPU. It is a separate workload and does not measure Newton's API or this box task.

## What the comparison measures

Both backends use a 330 × 250 × 55 mm receiving-box interior and separate placement targets. The gripper lowers each held cube before opening, with a nominal 32 mm gap above the floor. The receiving floor uses direct `solref = (-4444.444444, -266.666667)`, retaining the mixed normal stiffness and doubling normal damping. Other materials, friction coefficients and default collision routes remain unchanged. The original physical grasp, lift, carry, release and settling gates still apply.

`--task box` is the benchmark default. CPU and GPU receive the same compiled scene, initial state and precomputed float32 command tape from the canonical `BoxTask` controller. At reset, the arm starts at the first approach target above the red cube to avoid intersecting the table or box; both cubes remain at their original outside-box spawn positions. Each environment runs **2,000 control frames at 50 Hz**, with **20 physics steps per frame at 0.001 s**: **40,000 steps and 40 simulated seconds**. Keep that horizon fixed at every batch size, including the final settling and withdrawal period.

The CPU uses a persistent pool of worker **processes**, each advancing a native MuJoCo world and recording real contact evidence. `--cpu-baseline pool --cpu-threads 32` requests up to 32 workers, capped by the environment count. GPU execution batches the environments on one selected CUDA device. Native CPU physics uses float64 and GPU physics uses float32; both write the same float32 observation format and must pass the same task gate. Both backends use 100 solver iterations, 50 line-search iterations and `impratio = 100`. Reports preserve the native model settings and the actual uploaded GPU settings separately. Both backends request solver tolerance `1e-6` and line-search tolerance `0.01`. The benchmark retains each backend’s default collision route.

The simulation timer includes episode reset, command replay, physics stepping, the per-frame forward evaluation needed for current geometry and solved contact forces, and observation recording. CPU dispatch/completion and CUDA synchronization are included. Command generation, setup, compilation, warm-up, final host output collection and task validation are recorded separately. Compare simulation time and time to checked host results.

Both backends use the same requested host-validation budget (`--cpu-threads 32`), capped by world count and CPU affinity. One world validates in the parent; larger batches reuse CPU workers and read-only shared histories with CUDA hidden. Validation time includes history copy, dispatch, physical/clock checks and collection; worker startup and teardown are recorded separately.

At each control frame, both backends record **61 float32 fields per environment**: time, gripper opening and final-frame tool clearance (zero in earlier frames); then, for each cube, eight 3D corners, maximum corner speed, two jaw-contact counts and two solved jaw normal forces. This compact history supports physical grasp-to-release checks without recording every physics state. At 2,048 environments the history alone occupies **999,424,000 bytes (953.125 MiB)**; host copies and solver allocations require additional memory.

The comparison uses commands already available. Online inverse kinematics, rendering, policy training and the cloth-and-cable exercise are outside its scope.

## Task success

The same `CubeEvidence` rules apply to both cubes, in every world and every warm-up or measured repetition:

1. Grasp with contact on both jaws and solved normal force above **1e-5 N per jaw** for at least **10 consecutive frames**, establishing the grasp during closing.
2. Keep the whole cube at least **1 cm above the table** for **25 consecutive loaded frames**.
3. Retain the loaded grasp during at least **10 carry frames** and at least **5 cm of uninterrupted airborne XY travel**; all eight corners must reach over the box before opening.
4. Open the actual gripper to at least **80%**, then detach the cube from both jaws. Dropping it before release fails the task.
5. Keep all eight cube corners inside the receiving box, within **3 mm** tolerance, with maximum corner speed below **0.04 m/s** for **50 consecutive detached frames**.

The red cube must be released before the blue cube is grasped. At completion, both cubes must remain contained and settled, and the gripper must withdraw at least **2 cm above the higher of the table and box rim**. Finite observations, complete clock progression, all 40,000 actual physics steps and valid capacity checks are also required. The raw clock is checked against accumulation at the backend’s own precision; the float32 GPU clock can differ slightly from nominal elapsed time. An independent integration counter prevents a missing or duplicated step from passing that check. An object center inside the box, a phase label or a successful process exit is not sufficient evidence.

A failed or incomplete configuration has no accepted timing or speedup. The first failed episode stops that configuration; the report retains the completed episodes and the sweep continues with the other configurations. Preserve its raw evidence and failure status; do not shorten episodes, discard failing repetitions or substitute a smaller batch. MuJoCo Warp 3.8.0.3 uses per-step capacity high-water checks because newer sticky overflow flags are unavailable; the report retains that diagnostic distinction.

## Run the full sweep

Prepare the [Article 2 environment](README.md#reproduce-the-verified-environment), then run from `part2`:

```bash
python migration_benchmark.py --task box --cpu-baseline pool --preset workstation --device cuda:0 --cpu-threads 32 --nconmax 64 --njmax 128 --timeout 7200 --preflight
python migration_benchmark.py --task box --cpu-baseline pool --robot so101 \
  --worlds 1 16 32 64 128 256 512 1024 2048 --cpu-threads 32 --device cuda:0 \
  --nconmax 64 --njmax 128 --repeats 5 --warmups 1 --order-seed 0 \
  --timeout 7200 --max-trajectory-mib 8192 --output-dir .generated/box-full-so101
```

Requested box capacities are `nconmax=64` and `njmax=128`. Reports retain actual allocated limits and high-water counts; capacity overflow invalidates a case. The 7,200-second case timeout does not shorten the 40-second simulated episode.

Repeat with `--robot rebot` and a fresh directory. `--preset workstation` contains all nine sizes; `--preset developer` selects 1 and 16. Explicit `--worlds` overrides the preset. A smaller list is an exploratory subset, not a full sweep.

Preflight reports devices, configuration and observation-memory estimates without generating performance numbers. The 8 GiB history budget does not reserve RAM/VRAM or cover every model, solver and scratch allocation. Check memory, CPU allocation and concurrent applications before enabling measurements. `--device cuda:0` selects the first visible CUDA device; choose the available device on your machine. Each invocation uses one GPU. `--cpu-only` leaves GPU performance unmeasured.

The notebook starts with `RUN_BENCHMARK = False`, so Run All performs setup and preflight only. Review its printed command before enabling measurements.

## Historical stacking experiment

`--task stack` retains the earlier 600-frame, 12-second stacking protocol and its native C++ rollout baseline. Its observations and acceptance criteria differ from the box task. Keep those reports under their original task label and compare only compatible CPU/GPU rows. The existing stacking figures must not be reused as measurements of placing both cubes in a box.

## Save and interpret reports

Each run writes `results.json`, `summary.csv`, `summary.md` and raw worker reports/logs. Preserve the source revision, package lock, CPU/GPU/RAM, driver, environment counts, worker allocation, individual samples and every task outcome. Use a fresh output directory for each run; compiled caches are not measurement evidence.

Compare CPU and GPU at equal environment count and task duration. Report medians and the observed minimum–maximum across the five accepted repetitions, along with the actual CPU worker count. Include setup/compilation cost when it matters to the application. These community measurements describe the recorded machine and load; they do not establish performance on another workstation or cloud GPU.

One environment tests latency; a large batch tests throughput. GPU execution can take longer at small batch sizes even when a larger batch is faster. [MuJoCo’s MJWarp guidance](https://mujoco.readthedocs.io/en/latest/mjwarp/#low-latency) describes this distinction. The sweep measures whether and where a crossover occurs for this task; GPU acceleration is not a condition for accepting a valid result.

## Separate ALOHA reference benchmark

The [ALOHA reference benchmark](reference-benchmark/README.md) compares native MuJoCo with MuJoCo Warp on a separate reference workload. It measures neither Newton’s API path nor this two-cube receiving-box task. Keep its timings and hardware/configuration records separate from the box results.

## Google Colab

Open [the benchmark notebook on Colab](https://colab.research.google.com/github/johnnynunez/accelerated-computing-hub/blob/feature/blog2-validated-box-benchmark/tutorials/sim2real-blogs/notebooks/mujoco/part2/04__cpu_gpu_benchmark.ipynb) and select a GPU runtime. The setup checks out `REPO_URL` at `REF`, installs uv 0.12.5 into a tools environment, and syncs the hash-locked dependencies into a separate Python 3.12 environment. Benchmark subprocesses use that interpreter; the hosted kernel remains separate. `REF` selects the public Hub source; retain the resolved full commit printed by setup with any new results. A changed revision requires a fresh checkout directory. The pre-merge link uses the public contribution branch.

Review preflight on the assigned CPU/GPU before enabling measurements. Hosted hardware and runtime limits vary; this launch recipe is not a validated Colab box-task result. Download the report archive before disconnection. The CPU baseline belongs to the runtime.

## NVIDIA Brev

An existing Brev GPU VM can use the same repository and lock:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch feature/blog2-validated-box-benchmark https://github.com/johnnynunez/accelerated-computing-hub.git
cd accelerated-computing-hub
git rev-parse HEAD  # retain the resolved source revision
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes \
  tutorials/sim2real-blogs/notebooks/mujoco/requirements.lock.txt
.venv/bin/python tutorials/sim2real-blogs/notebooks/mujoco/part2/migration_benchmark.py \
  --task box --preset workstation --nconmax 64 --njmax 128 --timeout 7200 --preflight
```

Use that interpreter for the measurement command or notebook. Review the instance allocation and preserve reports before stopping the VM. Box-task validation on Brev is pending.
