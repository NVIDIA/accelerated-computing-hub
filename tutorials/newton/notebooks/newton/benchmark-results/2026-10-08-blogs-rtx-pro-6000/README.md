# Blog 3 task measurements — October 8, 2026

This earlier study covers one-world online Newton workflows and a limited replay sweep. Its tested source is [commit ca713d3](https://github.com/johnnynunez/blogs/tree/ca713d36159434bb5a7569115a11efffbc4a292d); the recorded source hashes describe that snapshot, not subsequent changes. The current [two-cube box benchmark](../../BENCHMARK.md) uses a different task and protocol; these earlier stacking measurements do not establish its performance.

**Newton's CPU workflow completed this single-world task faster than its CUDA workflow for both robots.** The separate replay sweep also found no GPU advantage over the fastest measured CPU configuration at the tested batch sizes. These are community observations for team review, not official NVIDIA product benchmarks or hardware recommendations.

## Newton CPU and CUDA

Each episode completes 12 simulated seconds of two-cube pick-and-place: 600 online controller updates and 6,000 physics steps, without rendering. Values below are medians of five full episodes after one full warm-up. The workflow timer includes IK, control/state transfers, stepping and payload recording. “Checked” additionally includes final host download and validation; construction, compilation and warm-up are separate.

| Robot | Newton CPU (s) | Newton CUDA (s) | CPU/CUDA ratio | Checked CPU/CUDA ratio |
|---|---:|---:|---:|---:|
| SO-101 | 1.6284 | 2.3388 | 0.696× | 0.697× |
| reBot | 1.5519 | 2.7662 | 0.561× | 0.562× |

Below 1× means CUDA took longer. Both Newton devices use matching source, assets, solver settings and observation protocol, including 100 solver iterations, 50 line-search iterations and `impratio=100`. Their compiled-model hashes differ and are preserved in the reports. These are application cost ratios; they do not establish identical compiled physics, batched Newton performance or solver-coupling performance.

The separate native MuJoCo application baseline took **0.2517 s** for SO-101 and **0.2626 s** for reBot. It retains the tutorial's different model/actuation construction: SO-101 uses 10/20 solver/line-search iterations, reBot 100/50, and both use `impratio=10`. It therefore supplies context rather than an identical-physics ratio against Newton.

## Native MuJoCo and MuJoCo Warp replay

Replay holds the compiled model, initial state and quantized control tape constant across backends, using 100/50 iterations and `impratio=100`. Timing includes reset and full-state recording after every substep: float64 in CPU RAM, float32 in GPU memory. Final download and validation are separately measured.

| Robot | Worlds | CPU workers | CPU (s) | GPU (s) | CPU/GPU: simulation / checked |
|---|---:|---:|---:|---:|---:|
| SO-101 | 1 | 1 | 0.2195 | 2.3671 | 0.093× / 0.093× |
| SO-101 | 16 | 16 | 0.2250 | 2.8058 | 0.080× / 0.083× |
| SO-101 | 64 | 32 | 0.4320 | 3.0428 | 0.142× / 0.150× |
| SO-101 | 256 | 32 | 1.5785 | 3.4175 | 0.462× / 0.472× |
| reBot | 1 | 1 | 0.1818 | 2.9119 | 0.062× / 0.063× |
| reBot | 16 | 16 | 0.1881 | 3.5158 | 0.054× / 0.056× |

The table selects the fastest measured CPU configuration per world count; the same CPU case won both timing scopes. At 256 SO-101 worlds, **one CPU worker took 49.0243 s**, compared with **3.4175 s on GPU** (14.345×). With **32 CPU workers**, CPU time fell to **1.5785 s** (0.462×). Reporting the CPU baseline changes the interpretation. No crossover outside this sweep was measured.

## Hardware, validation and provenance

The machine had an AMD Ryzen Threadripper PRO 9985WX (**64 physical cores / 128 logical CPUs**) and **264,948,391,936 bytes RAM** (246.75 GiB visible). Runs selected `cuda:1`, an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, reporting **97,887 MiB VRAM** and driver **615.71.09**. A second RTX PRO 6000 was present; each run used one GPU.

Recorded software: Linux 7.0.0-38-generic x86-64; Python 3.12.13; Newton 1.6.0; MuJoCo / MuJoCo Warp 3.12.0; Warp 1.17.0; NumPy 2.5.3. The machine was shared: GPU 0 had other active workloads, and a resident `llama-server` occupied approximately 22,328 MiB on GPU 1. Before/after load snapshots do not establish exclusive use throughout a run.

All **22 cases passed**: **132 complete batch episodes**, including warm-ups, representing **6,396 individual world episodes**. The six workflow cases account for 36 episodes. Validation requires airborne pickup/continuous transport, stacking on the table, finite state, complete duration, settling and available capacity. See the [measurement methods and acceptance criteria](../../BENCHMARK.md). No lower-end GPU, Colab runtime or live Brev deployment was validated here.

SO-101 replay retains `LS_ITERATIONS` flags in 62 GPU world-episode records, including warm-ups. These indicate a line-search iteration limit; all task gates passed and no capacity overflow occurred. The reports preserve these diagnostics and do not establish full solver convergence. The Newton workflows and reBot replay had no such flags.

The remote execution used a copy of the canonical `johnnynunez/blogs` sources without `.git`; Git fields are therefore null. All four reports have verified per-file hashes and source aggregate `9e2e8eb5e79a4fc0374b0ad7750496dc2f77d08e10c8ea03ee929471c88efe92`. Individual timings, model hashes and diagnostic details remain in the raw reports:

- SO-101 Newton workflow: [JSON](workflow-so101/results.json), [CSV](workflow-so101/summary.csv), [report](workflow-so101/summary.md).
- reBot Newton workflow: [JSON](workflow-rebot/results.json), [CSV](workflow-rebot/summary.csv), [report](workflow-rebot/summary.md).
- SO-101 replay: [compressed JSON](replay-so101/results.json.gz), [CSV](replay-so101/summary.csv), [report](replay-so101/summary.md).
- reBot replay: [compressed JSON](replay-rebot/results.json.gz), [CSV](replay-rebot/summary.csv), [report](replay-rebot/summary.md).

The [manifest](manifest.json) records artifact digests. To reproduce this earlier study, check out the tested commit above and run from `Article_3/part3` using its locked Article 3 environment, with fresh output directories:

```bash
python migration_benchmark.py --scope workflow --robot so101 --device cuda:1 \
  --repeats 5 --warmups 1 --output-dir .generated/workflow-so101
python migration_benchmark.py --scope workflow --robot rebot --device cuda:1 \
  --repeats 5 --warmups 1 --output-dir .generated/workflow-rebot
python migration_benchmark.py --robot so101 --worlds 1 16 64 256 --cpu-threads 32 \
  --device cuda:1 --repeats 5 --warmups 1 --output-dir .generated/replay-so101
python migration_benchmark.py --robot rebot --worlds 1 16 --cpu-threads 32 \
  --device cuda:1 --repeats 5 --warmups 1 --output-dir .generated/replay-rebot
```

Select the device available on your machine. The CPU pool is capped at the world count and available CPU affinity; the 16-world cases use at most 16 workers.
