# Task measurements on RTX PRO 6000 — October 8, 2026

**GPU execution did not beat the fastest measured CPU configuration at any tested batch size** for this two-cube task: SO-101 used 1, 16, 64 and 256 worlds; reBot used 1 and 16.

These community measurements are candidates for team review, not official NVIDIA product benchmarks. They describe the specified workstation and workload. No developer laptop GPU, lower-end GPU, live Brev deployment or Colab runtime was validated by these runs.

## Hardware and configuration

| Component | Recorded configuration |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 9985WX, 64 physical cores / 128 logical CPUs |
| System memory | 264,948,391,936 bytes visible to Linux, approximately 246.75 GiB |
| Selected GPU | `cuda:1`, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB reported VRAM |
| Other detected GPU | A second RTX PRO 6000 at index 0; these measurements use GPU 1 only |
| Driver | 615.71.09 |
| OS | Linux x86-64, kernel 7.0.0-38-generic |
| Python and packages | Python 3.12.13; MuJoCo / MuJoCo Warp 3.12.0; Warp 1.17.0; Newton 1.6.0; NumPy 2.5.3 |
| Episode | 12 simulated seconds, 600 control frames at 50 Hz, 6,000 physics steps at 0.002 s |
| Sampling | Five measured full episodes after one full warm-up episode per case |

This was a shared machine. A resident `llama-server` process occupied **22,328 MiB** of GPU memory in the recorded snapshots. Before/after case snapshots retain CPU load, GPU utilization, memory, temperature, clocks and power; they do not establish exclusive use throughout the measurement. No service was stopped to produce an isolated-hardware claim.

## Identical-model replay

Both backends replay the same compiled MJCF configuration and quantized control tape, with 100 solver iterations, 50 line-search iterations and `impratio = 100`. The simulation timer includes reset, stepping and full-state recording after every substep. CPU records float64 state in RAM; GPU records float32 state in VRAM. “Checked” includes host transfer and task validation, while setup and warm-up remain separate.

Each cell below shows **simulation / checked-result medians**. The CPU column uses the fastest measured CPU configuration at that world count. Speedup is CPU time divided by GPU time; **less than 1× means the GPU was slower**. The same CPU case was fastest for both timing scopes in these reports.

| Robot | Worlds | CPU workers | CPU simulation / checked (s) | GPU simulation / checked (s) | GPU speedup: simulation / checked |
|---|---:|---:|---:|---:|---:|
| SO-101 | 1 | 1 | 0.2174 / 0.2184 | 2.3694 / 2.3707 | 0.092× / 0.092× |
| SO-101 | 16 | 16 | 0.2001 / 0.2073 | 2.7977 / 2.8044 | 0.072× / 0.074× |
| SO-101 | 64 | 32 | 0.4530 / 0.4830 | 3.0477 / 3.0868 | 0.149× / 0.156× |
| SO-101 | 256 | 32 | 1.5937 / 1.6975 | 3.4102 / 3.5581 | 0.467× / 0.477× |
| reBot | 1 | 1 | 0.1892 / 0.1901 | 2.9153 / 2.9167 | 0.065× / 0.065× |
| reBot | 16 | 16 | 0.1900 / 0.1989 | 3.5142 / 3.5228 | 0.054× / 0.056× |

All **16 replay cases** passed, including every world in all five measured batches and the warm-up batch. No GPU advantage over the fastest measured CPU configuration appeared at the tested sizes. This does not establish a crossover outside the sweep or predict performance on another CPU/GPU.

The CPU baseline changes the interpretation. At 256 SO-101 worlds, one CPU worker took **42.3698 s**, so comparison against that row alone gives the GPU a **12.42×** simulation speedup. The measured 32-worker CPU pool took **1.5937 s**, faster than the GPU's **3.4102 s**. Both CPU configurations remain in the reports; a single-thread-only headline would omit the stronger baseline.

## Single-world workflow

These runs measure the actual scripted applications: online waypoint IK, control updates, physics and both cubes' pose/velocity recording at 50 Hz. The simulation timer includes recording in RAM for CPU paths or VRAM for CUDA. “Checked result” adds final host transfer and task validation. Fresh model/controller construction, compilation and warm-up are reported separately and excluded from these columns.

| Robot | Workflow | Median simulation (s) | Min–max (s) | Median checked result (s) | Measured episodes passed |
|---|---|---:|---:|---:|---:|
| SO-101 | Native MuJoCo, CPU | 0.2817 | 0.2798–0.2830 | 0.2835 | 5/5 |
| SO-101 | Newton, native MuJoCo CPU | 1.5878 | 1.4916–1.6758 | 1.5896 | 5/5 |
| SO-101 | Newton, MuJoCo Warp GPU | 2.3273 | 2.3227–2.3289 | 2.3291 | 5/5 |
| reBot | Native MuJoCo, CPU | 0.2601 | 0.2597–0.2625 | 0.2639 | 5/5 |
| reBot | Newton, native MuJoCo CPU | 1.5358 | 1.3892–1.5515 | 1.5397 | 5/5 |
| reBot | Newton, MuJoCo Warp GPU | 2.7489 | 2.7301–2.7691 | 2.7525 | 5/5 |

Each workflow has one world and one CPU rollout thread. The top-level replay world/worker defaults remain in the configuration record but do not apply to workflow cases; each case records its actual settings. All six warm-up episodes also passed, for **36 successful complete workflow episodes** across the six cases.

Newton reconstructs the scene and actuation. The native SO-101 workflow uses its source settings of 10 solver / 20 line-search iterations; the native reBot workflow uses 100 / 50. Both native workflows use `impratio = 10`, while the Newton workflows use 100 / 50 and `impratio = 100`. These application costs therefore cannot establish an identical-model CPU/GPU speedup or a batch crossover. They show the cost of these particular one-world migrations, including the host controller and different model/solver configuration.

The shared task gate checks whole-cube lift, continuous airborne carry, placement on the supported blue cube and sustained settling. Settling uses the final-second RMS point-speed bound with unsmoothed position/orientation limits; raw velocity peaks remain available. Every recorded Newton CUDA workflow episode had zero capacity-overflow and iteration-warning flags. See the [benchmark methodology](../../BENCHMARK.md) for exact thresholds and timing boundaries.

## Reports and reproducibility

- SO-101 replay: [complete JSON, gzip](replay-so101/results.json.gz), [CSV](replay-so101/summary.csv), [generated summary](replay-so101/summary.md).
- reBot replay: [complete JSON, gzip](replay-rebot/results.json.gz), [CSV](replay-rebot/summary.csv), [generated summary](replay-rebot/summary.md).
- SO-101 workflow: [full JSON](workflow-so101/results.json), [CSV](workflow-so101/summary.csv), [generated summary](workflow-so101/summary.md).
- reBot workflow: [full JSON](workflow-rebot/results.json), [CSV](workflow-rebot/summary.csv), [generated summary](workflow-rebot/summary.md).

The full JSON files contain every measured and warm-up sample, task outcomes, model/control/source identities, solver settings, timing components and load snapshots. They were collected from a copied source tree without Git metadata, so `git_commit` is `null`. The source-file SHA-256 values bind the code actually used; neither a missing commit nor a mutable branch name is presented as an immutable Git revision. All four reports record the source aggregate `5f43ddbb7afe812ff9d8fda7aa9138ba04c006eba22ec2809714e5ee53e7a73b`.

The replay JSON files are compressed with gzip, preserving the original complete JSON bytes and every unrounded sample. [manifest.json](manifest.json) records both compressed-file and decompressed-JSON SHA-256 hashes, case counts and the source aggregate. Use `gzip.open(path, "rt")` with Python's `json.load` to inspect them, or decompress them with a standard gzip tool.

From `tutorials/newton/notebooks/newton/part3`, use the pinned environment and fresh output directories:

```bash
python migration_benchmark.py --scope replay --robot so101 --worlds 1 16 64 256 \
  --cpu-threads 32 --device cuda:1 --repeats 5 --warmups 1 \
  --output-dir .generated/replay-so101
python migration_benchmark.py --scope replay --robot rebot --worlds 1 16 \
  --cpu-threads 16 --device cuda:1 --repeats 5 --warmups 1 \
  --output-dir .generated/replay-rebot
python migration_benchmark.py --scope workflow --robot so101 --device cuda:1 \
  --repeats 5 --warmups 1 --output-dir .generated/workflow-so101
python migration_benchmark.py --scope workflow --robot rebot --device cuda:1 \
  --repeats 5 --warmups 1 --output-dir .generated/workflow-rebot
```

Select the GPU actually available on another machine, preserve all resulting metadata and rerun the full episodes. These workstation observations require independent validation on the developer-accessible GPU selected for the article.
