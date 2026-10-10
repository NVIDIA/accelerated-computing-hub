# Article 2: full two-cube box CPU/GPU sweep

Native MuJoCo CPU versus MuJoCo Warp, measured at **1, 16, 32, 64, 128, 256, 512, 1024 and 2048 worlds** for both robots. These are community task measurements on shared hardware, not official product benchmarks or a minimum hardware specification. All **36 requested cases** are retained: **29 passed**, **7 failed or incomplete**. A failed size has no accepted timing median or CPU/GPU ratio.

## Scaling figures

Both figures show the median and observed min–max of five accepted repetitions after one warm-up. Failed configurations are untimed gaps.

![CPU/GPU simulation batch time](figures/fig_box_cpu_gpu_scaling.png)

[Vector PDF](figures/fig_box_cpu_gpu_scaling.pdf) · [SVG](figures/fig_box_cpu_gpu_scaling.svg) · [Full caption](figures/fig_box_cpu_gpu_scaling.caption.txt)

![CPU/GPU time to checked host results](figures/fig_box_cpu_gpu_checked_results.png)

Checked host time adds simulation, host output collection/download and task validation within each repetition. Setup, warm-up and validator teardown remain separate. [Vector PDF](figures/fig_box_cpu_gpu_checked_results.pdf) · [SVG](figures/fig_box_cpu_gpu_checked_results.svg) · [Full caption](figures/fig_box_cpu_gpu_checked_results.caption.txt).

The [plotted data](figures/fig_box_cpu_gpu_scaling.data.json) retain raw per-repetition component times and exclusions. The [standalone renderer](figures/gen_fig_box_cpu_gpu_scaling.py) reproduces both figures from this package's compressed reports using optional Matplotlib, without changing the physics environment.

## Validated timings

Values are median seconds for five complete 40-second episodes after one full warm-up. Protocol v2 uses 2,000 control frames, each containing 20 integration steps of 0.001 s: 40,000 steps per world. CPU/GPU is CPU time divided by GPU time; below 1× means the GPU took longer. The CPU pool uses up to 32 workers, capped by world count. This sweep measures that pool baseline; it does not supply a separate serial-CPU comparison at every size.

| Worlds | CPU workers | SO-101 CPU (s) | SO-101 GPU (s) | CPU/GPU | reBot CPU (s) | reBot GPU (s) | CPU/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.3000 | 14.6314 | 0.089× | 1.0518 | 18.6417 | 0.056× |
| 16 | 16 | 1.3266 | 16.7440 | 0.079× | 1.3047 | 21.0475 | 0.062× |
| 32 | 32 | 1.3482 | 18.1925 | 0.074× | 1.2857 | 22.5130 | 0.057× |
| 64 | 32 | 2.6656 | 19.5688 | 0.136× | 2.5379 | 23.7271 | 0.107× |
| 128 | 32 | 5.2321 | 21.4587 | 0.244× | 5.0683 | **failed** | — |
| 256 | 32 | 10.4544 | 23.4024 | 0.447× | 10.3190 | **failed** | — |
| 512 | 32 | 21.1330 | **failed** | — | 20.2056 | 30.1455 | 0.670× |
| 1024 | 32 | 41.0932 | **failed** | — | 39.8046 | **failed** | — |
| 2048 | 32 | 81.7012 | **failed** | — | 79.1299 | **failed** | — |

SO-101: first accepted tested size with faster GPU simulation **none**; with checked host results **none**; reBot: first accepted tested size with faster GPU simulation **none**; with checked host results **none**. These are observations at the tested sizes, not universal crossover thresholds; failed cases are excluded.

CPU advances native MuJoCo worlds in persistent worker processes; GPU batches the same compiled scene and recorded targets through MuJoCo Warp. Both record 61 float32 values per world at each of 2,000 control frames: time, actual gripper opening, final jaw clearance, both cubes’ corners and speeds, and actual solved bilateral jaw contacts and forces. Both paths use 100 solver iterations, 50 line-search iterations and `impratio=100`. Timers include episode reset, stepping and observation recording, with CUDA synchronization or CPU worker dispatch/completion. Setup, warm-up, host output collection and validation are recorded separately. The raw reports also give timings through checked host results. Article 2 and Article 3 use different dependency versions and model conversion paths; their tables do not establish a cross-article speedup. See [methods and acceptance criteria](../../BENCHMARK.md).

Execution revision v3 applies the unchanged serial validation oracle independently to each world. CPU and GPU cases use the same host-validation worker budget, capped by world count and CPU affinity. One worker runs in the parent; larger batches reuse a spawned pool and read-only shared histories. Validation time includes the history copy, dispatch, all checks and result collection. Pool setup and teardown are recorded separately and excluded from per-episode simulation and checked-result timings. The audit retains requested/actual workers, shared-history size and initialization/cleanup costs.

## Task outcomes

Every accepted case passed in every world during all six episodes: the complete 40-second duration, finite state and time progression, loaded bilateral grasps, whole-cube lifts, retained airborne carry, intentional release, both cubes fully contained and detached with corner speeds below 0.04 m/s for at least one second, final gripper clearance of 2 cm, and capacity checks. No cube may be dragged or dropped before release. These are box measurements; earlier stacking runs are a different protocol. Failed cases stop at the first rejected episode; earlier successful repetitions do not make that case eligible.

- SO-101, mjwarp, 512 worlds: red_cube: settling incomplete (2 world-episodes). Recorded episode counts: 1 warm-up, 0 measured; excluded from timing ratios.
- SO-101, mjwarp, 1024 worlds: red_cube: settling incomplete (1 world-episode). Recorded episode counts: 1 warm-up, 0 measured; excluded from timing ratios.
- SO-101, mjwarp, 2048 worlds: red_cube: settling incomplete (1 world-episode); blue_cube: settling incomplete (1 world-episode). Recorded episode counts: 1 warm-up, 0 measured; excluded from timing ratios.
- reBot, mjwarp, 128 worlds: red_cube: settling incomplete (1 world-episode). Recorded episode counts: 1 warm-up, 2 measured; excluded from timing ratios.
- reBot, mjwarp, 256 worlds: red_cube: settling incomplete (1 world-episode). Recorded episode counts: 1 warm-up, 0 measured; excluded from timing ratios.
- reBot, mjwarp, 1024 worlds: red_cube: settling incomplete (1 world-episode). Recorded episode counts: 1 warm-up, 0 measured; excluded from timing ratios.
- reBot, mjwarp, 2048 worlds: blue_cube: settling incomplete (1 world-episode); red_cube: settling incomplete (1 world-episode). Recorded episode counts: 1 warm-up, 0 measured; excluded from timing ratios.

The [audit](audit.json) retains failed-world values and verifies count coverage, source/model/control signatures, every accepted warm-up/repetition, and exclusion from ratios. Its episode counts include all recorded attempts, including failed episodes; omitted later repetitions are not imputed. Every world supplies an independent integration count and a raw-clock receipt: all recorded timestamps must match sequential accumulation in the backend precision within one float32 observation ULP. Raw timestamps are retained. Actual native and uploaded GPU options are recorded separately, including the GPU tolerance clamp and float32 representation. Accepted CUDA runs also require raw contact, constraint and broadphase high-water checks after integration and observation-forward calls.

## Earlier preflight evidence

The separate [frozen preflight bundle](preflight-v3-evidence.tar.gz) retains all eight correctness reports across both articles: **15 passing cases and one excluded case**. Blog 3 SO-101 CUDA at 128 worlds passed its warm-up, then failed one measured world: world 76's blue cube recorded **27 terminal settled frames**, below the unchanged requirement of **50**. Counts, clocks and capacity checks passed. This is an observed settling failure; its transient cause was not established. See the [preflight summary](preflight-summary.json) and [detailed failure audit](preflight-failure-audit.json).

These preflights use one warm-up and one measured episode and are excluded from the full-sweep tables, charts and ratios. Their original sources and outcomes remain immutable. A later independent passing case does not replace this earlier failure; the full-sweep outcomes and source identities are reported separately.


## Online teaching checks

Both original online GPU teaching examples—SO-101 and reBot—also passed one complete 2,000-frame box episode on the workstation. The separate [teaching validation evidence](tutorial-validation/README.md) retains their exact commands, execution status, logs, original task reports and [source/file hashes](tutorial-validation/manifest.json). These are correctness checks, excluded from benchmark timings and ratios.

## Hardware and provenance

CPU: **AMD Ryzen Threadripper PRO 9985WX 64-Cores**, 64 physical cores / 128 logical CPUs; RAM **264,948,404,224 bytes** (246.75 GiB visible). Selected GPU: **cuda:1, NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, 97,887 MiB VRAM, driver **615.71.09**. The second GPU is not combined with the selected device.

Linux 7.0.0-38-generic x86_64; Python 3.12.13; mujoco 3.8.0, mujoco-warp 3.8.0.3, warp-lang 1.15.0, numpy 2.5.3. Other workloads and resident GPU allocations remained on the machine. Before/after utilization, memory and process snapshots document shared use; they do not prove exclusive hardware access throughout each episode. No lower-end GPU, Colab runtime or live Brev deployment was validated by these runs.

Both reports have source aggregate `8d39e63de5e991f6bb83e10050783bbdd4658786a3899e1633ca8413bbcf807e`. All recorded source-file hashes were checked against this article's source tree before packaging. The [executed source archive](source.tar.gz) retains canonical `Article_2/…` paths; every archived file was byte-verified against the checkout, including dependency files. The remote snapshot omitted `.git`, so source hashes identify the executed files. [Manifest](manifest.json) records source, archive-member and compressed/uncompressed report digests; gzip decompression restores the exact original JSON bytes.

- SO-101: [raw JSON.gz](so101/results.json.gz), [CSV](so101/summary.csv), [report](so101/summary.md), [complete run files](so101/raw-run.tar.gz).
- reBot: [raw JSON.gz](rebot/results.json.gz), [CSV](rebot/summary.csv), [report](rebot/summary.md), [complete run files](rebot/raw-run.tar.gz).

Run archives preserve preflight, every worker configuration, individual JSON reports and logs, including failures. Compiled `.warp-cache` files are excluded.

From `Article_2/part2` in its locked environment, reproduce each robot in a fresh output directory; choose your available CUDA device:

```bash
python migration_benchmark.py --task box --cpu-baseline pool --robot so101 \
  --worlds 1 16 32 64 128 256 512 1024 2048 --cpu-threads 32 --device cuda:1 \
  --repeats 5 --warmups 1 --max-trajectory-mib 8192 --timeout 7200 --output-dir .generated/box-full-so101
```

Repeat with `--robot rebot` and `.generated/box-full-rebot`. The 8 GiB trajectory limit excludes additional model/solver memory. Preserve failed rows and revalidate on the developer hardware of interest before drawing broader conclusions.
