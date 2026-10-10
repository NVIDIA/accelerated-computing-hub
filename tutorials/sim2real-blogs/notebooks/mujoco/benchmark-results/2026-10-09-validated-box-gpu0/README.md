# Article 2: matched-tolerance two-cube box benchmark

## When is migration worth it?

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

Use the [benchmark notebook](https://github.com/johnnynunez/blogs/blob/feature/blog2-validated-gpu-box/Article_2/part2/02_Notebook_CPU_GPU_Benchmark.ipynb) to reproduce the task. The [companion benchmark guide](https://github.com/johnnynunez/blogs/blob/feature/blog2-validated-gpu-box/Article_2/BENCHMARK.md) explains the measurement method; [full results](https://github.com/johnnynunez/blogs/blob/feature/blog2-validated-gpu-box/Article_2/benchmark-results/2026-10-09-validated-box-gpu0/README.md) retain ratios, ranges, failures and hardware configuration. These community measurements are not official product benchmarks or evidence of lower-end GPU, Colab or Brev performance.

The companion also includes an [ALOHA pot-and-lid replay](https://github.com/johnnynunez/blogs/blob/feature/blog2-validated-gpu-box/Article_2/reference-benchmark/README.md) comparing native MuJoCo on CPU with MuJoCo Warp on GPU. It is a separate workload and does not measure Newton's API or this box task.


## Full simulation medians and ratios

| Worlds | CPU workers | SO-101 CPU (s) | SO-101 GPU (s) | CPU/GPU | reBot CPU (s) | reBot GPU (s) | CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.2682 | 15.6355 | 0.081× | 1.1887 | 19.4162 | 0.061× |
| 16 | 16 | 1.4631 | 17.9851 | 0.081× | 1.3369 | 21.5820 | 0.062× |
| 32 | 32 | 1.4754 | 19.3022 | 0.076× | 1.3536 | 23.0548 | 0.059× |
| 64 | 32 | 2.9472 | 20.5499 | 0.143× | 2.6919 | 24.0782 | 0.112× |
| 128 | 32 | 5.8159 | 22.6108 | 0.257× | 5.3933 | 25.4998 | 0.212× |
| 256 | 32 | 11.7220 | 24.0796 | 0.487× | 10.6702 | 27.3368 | 0.390× |
| 512 | 32 | 23.2236 | 25.7572 | 0.902× | 21.1583 | Excluded | — |
| 1024 | 32 | 45.8203 | 29.2140 | 1.568× | 42.0190 | 32.0479 | 1.311× |
| 2048 | 32 | 90.2232 | 34.4835 | 2.616× | 83.1366 | 38.1011 | 2.182× |

## Full checked-result medians and ratios

| Worlds | CPU workers | SO-101 CPU (s) | SO-101 GPU (s) | CPU/GPU | reBot CPU (s) | reBot GPU (s) | CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.3603 | 15.7199 | 0.087× | 1.2734 | 19.5007 | 0.065× |
| 16 | 16 | 1.5674 | 18.0901 | 0.087× | 1.4365 | 21.6840 | 0.066× |
| 32 | 32 | 1.6021 | 19.4082 | 0.083× | 1.4761 | 23.1939 | 0.064× |
| 64 | 32 | 3.1564 | 20.7623 | 0.152× | 2.9046 | 24.2924 | 0.120× |
| 128 | 32 | 6.2248 | 23.1135 | 0.269× | 5.8149 | 25.9927 | 0.224× |
| 256 | 32 | 12.5763 | 24.9939 | 0.503× | 11.4859 | 28.3091 | 0.406× |
| 512 | 32 | 24.8381 | 27.4834 | 0.904× | 22.7821 | Excluded | — |
| 1024 | 32 | 49.0296 | 32.6168 | 1.503× | 45.2637 | 35.3216 | 1.281× |
| 2048 | 32 | 96.7232 | 41.0464 | 2.356× | 89.5729 | 44.7091 | 2.003× |


## Methods and limits

The executed revision uses a wider, shallow receiving box (330 × 250 × 55 mm) with separated placement targets. The gripper lowers each held cube before release; the nominal placement gap is 32 mm so the held cube retains whole-corner lift clearance despite its orientation. The receiving floor uses direct solref (-4444.444444, -266.666667): the effective mixed normal stiffness is retained and normal damping is doubled. This is an authored contact-model change, not a timing adjustment. Other materials, friction coefficients and the default collision policy remain unchanged. Earlier geometries, floor trials and their failures remain separate historical evidence.

Both CPU and GPU request solver tolerance 1e-6, line-search tolerance .01, iteration limits 100/50 and impratio=100. Actual backend option values are stored separately, including float32 representations. This avoids giving the native CPU a stricter stopping target than the GPU’s supported tolerance floor. CPU integration is float64 and GPU integration is float32; both record the same 61 float32 physical observations at 50 Hz. Every raw timestamp is checked against sequential accumulation at the actual backend precision within one observation ULP, alongside an independently counted 40000 integration calls. Raw clock values are never rewritten.

CPU and GPU use the same requested 32-worker host-validation budget, capped by world count and CPU affinity. One world validates in the parent; larger batches reuse spawned CPU workers with read-only shared histories and CUDA hidden. Validation time includes history copy, dispatch, all physical/clock checks and collection. Startup and teardown are measured separately. No physics or acceptance gate is removed from the reported checked-result metric.


This package contains all 36 requested cases, including failed, timed-out or unavailable cases. All 18 configurations per robot must have a terminal result before packaging; accepted cases require one passing warm-up and five passing measurements in every world. Failed cases retain their raw timings for diagnosis but have no accepted median or ratio. Min/max ranges describe the five measured repetitions, not confidence intervals. Timers include reset, controls, 40000 steps and coherent state/solved-contact recording; IK preparation, model setup and compilation are separate. Earlier numerical trials, stress tests and historical full sweeps cannot supply missing repetitions here.

## Hardware and reproducibility

Shared CPU: **AMD Ryzen Threadripper PRO 9985WX 64-Cores**. Selected GPU: **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, **cuda:0**. Visible RAM: **264,948,404,224 bytes**. Versions: mujoco 3.8.0, mujoco-warp 3.8.0.3, warp-lang 1.15.0, numpy 2.5.3. Complete hardware, driver, CPU affinity, before/after process/load and memory snapshots remain in the raw reports. These snapshots do not establish exclusive access throughout each episode. The second GPU is not combined with the selected device.

Source aggregate: `dbcc514fd486889751dc40b880691fd46d319282fbabd5cd7c10be462362619f`. [Source archive](source.tar.gz) preserves all 64 files of the executed study, including both article stacks, under `sources/articleN/Article_N/`. [Source manifest](source-manifest.json) records their canonical relative paths and both report identities. [Plan](executed-plan.json) preserves the exact launch arguments. The manifest below covers every generated artifact and both compressed and original report hashes. The remote source copy has no Git metadata; file hashes identify what ran.

- SO-101: [JSON.gz](so101/results.json.gz), [CSV](so101/summary.csv), [Markdown](so101/summary.md), [run files](so101/raw-run.tar.gz).
- reBot: [JSON.gz](rebot/results.json.gz), [CSV](rebot/summary.csv), [Markdown](rebot/summary.md), [run files](rebot/raw-run.tar.gz).
- [Audit and failed-world evidence](audit.json), [article section](article-section.md), [structured section](article-section.json), [methods](methodology.md), [artifact manifest](manifest.json).

Gzip decompression restores the original JSON bytes. Run archives retain worker configurations, logs, individual results, initial report and hardware evidence, excluding compiled `.warp-cache` and symlinked cache directories. Ordinary CLI runs retain every world’s verdict and clock/count/capacity receipts; they do not retain the full temporal observation arrays. Separate motion and stress diagnostics retain those larger temporal arrays in the local/remote evidence archive; those multi-gigabyte histories are not bundled here or counted as repetitions in this study. Their one-warm-up/one-measurement checks are not final performance results.

From `Article_2/part2` in the pinned environment, use a new output directory:

```bash
python migration_benchmark.py --task box --cpu-baseline pool --robot so101 \
  --worlds 1 16 32 64 128 256 512 1024 2048 --cpu-threads 32 --device cuda:0 \
  --nconmax 64 --njmax 128 --repeats 5 --warmups 1 --order-seed 0 \
  --timeout 7200 --max-trajectory-mib 8192 --output-dir .generated/box-full-so101
```

Repeat for `--robot rebot` with another fresh output directory; select your available CUDA device. Revalidate all capacity and task checks on your hardware. The trajectory limit excludes additional model and solver memory. ALOHA/reference replay results belong to their own workload and package; no cross-workload or cross-article ratio is reported here.
