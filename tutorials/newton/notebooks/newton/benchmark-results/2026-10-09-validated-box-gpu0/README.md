# Article 3: matched-tolerance two-cube box benchmark

## When is migration worth it?

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

Use the [benchmark notebook](https://github.com/johnnynunez/blogs/blob/feature/blog3-validated-gpu-box/Article_3/part3/05_Notebook_Migration_Benchmark.ipynb) to reproduce the task. The [companion benchmark guide](https://github.com/johnnynunez/blogs/blob/feature/blog3-validated-gpu-box/Article_3/BENCHMARK.md) explains the measurement method; [full results](https://github.com/johnnynunez/blogs/blob/feature/blog3-validated-gpu-box/Article_3/benchmark-results/2026-10-09-validated-box-gpu0/README.md) retain ratios, ranges, failures and hardware configuration. These community measurements are not official product benchmarks or evidence of lower-end GPU, Colab or Brev performance.

The companion also includes an [ALOHA pot-and-lid replay](https://github.com/johnnynunez/blogs/blob/feature/blog3-validated-gpu-box/Article_3/reference-benchmark/README.md) comparing native MuJoCo on CPU with MuJoCo Warp on GPU. It is a separate workload and does not measure Newton's API or this box task.


## Full simulation medians and ratios

| Worlds | CPU workers | SO-101 CPU (s) | SO-101 GPU (s) | CPU/GPU | reBot CPU (s) | reBot GPU (s) | CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 6.4848 | 14.7928 | 0.438× | 6.3294 | 15.5382 | 0.407× |
| 16 | 16 | 7.2303 | 16.7782 | 0.431× | 7.4193 | 17.1097 | 0.434× |
| 32 | 32 | 7.6543 | 17.2842 | 0.443× | 7.4869 | 17.6607 | 0.424× |
| 64 | 32 | 15.2200 | 17.8851 | 0.851× | 15.1986 | 18.6563 | 0.815× |
| 128 | 32 | 30.4011 | 18.7815 | 1.619× | 29.9407 | 19.3498 | 1.547× |
| 256 | 32 | 60.7351 | 19.9407 | 3.046× | 59.7683 | 20.6777 | 2.890× |
| 512 | 32 | 122.6874 | 21.6250 | 5.673× | 119.8159 | 22.3135 | 5.370× |
| 1024 | 32 | 242.4549 | 23.8462 | 10.167× | 238.9082 | Excluded | — |
| 2048 | 32 | 482.4144 | 28.3054 | 17.043× | 475.6768 | 30.0213 | 15.845× |

## Full checked-result medians and ratios

| Worlds | CPU workers | SO-101 CPU (s) | SO-101 GPU (s) | CPU/GPU | reBot CPU (s) | reBot GPU (s) | CPU/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 6.5694 | 14.8775 | 0.442× | 6.4140 | 15.6225 | 0.411× |
| 16 | 16 | 7.3300 | 16.8853 | 0.434× | 7.5236 | 17.2137 | 0.437× |
| 32 | 32 | 7.7816 | 17.4233 | 0.447× | 7.6047 | 17.7798 | 0.428× |
| 64 | 32 | 15.4862 | 18.1545 | 0.853× | 15.4383 | 18.8924 | 0.817× |
| 128 | 32 | 30.8513 | 19.2337 | 1.604× | 30.3586 | 19.8185 | 1.532× |
| 256 | 32 | 61.5408 | 20.7737 | 2.962× | 60.5753 | 21.5811 | 2.807× |
| 512 | 32 | 124.3718 | 23.2860 | 5.341× | 121.4300 | 23.9868 | 5.062× |
| 1024 | 32 | 245.6746 | 27.1328 | 9.055× | 242.1689 | Excluded | — |
| 2048 | 32 | 488.8562 | 34.9235 | 13.998× | 482.1167 | 36.5507 | 13.190× |


## Methods and limits

The executed revision uses a wider, shallow receiving box (330 × 250 × 55 mm) with separated placement targets. The gripper lowers each held cube before release; the nominal placement gap is 32 mm so the held cube retains whole-corner lift clearance despite its orientation. The receiving floor uses direct solref (-4444.444444, -266.666667): the effective mixed normal stiffness is retained and normal damping is doubled. This is an authored contact-model change, not a timing adjustment. Other materials, friction coefficients and the default collision policy remain unchanged. Earlier geometries, floor trials and their failures remain separate historical evidence.

Both CPU and GPU request solver tolerance 1e-6, line-search tolerance .01, iteration limits 100/50 and impratio=100. Actual backend option values are stored separately, including float32 representations. This avoids giving the native CPU a stricter stopping target than the GPU’s supported tolerance floor. CPU integration is float64 and GPU integration is float32; both record the same 61 float32 physical observations at 50 Hz. Every raw timestamp is checked against sequential accumulation at the actual backend precision within one observation ULP, alongside an independently counted 40000 integration calls. Raw clock values are never rewritten.

CPU and GPU use the same requested 32-worker host-validation budget, capped by world count and CPU affinity. One world validates in the parent; larger batches reuse spawned CPU workers with read-only shared histories and CUDA hidden. Validation time includes history copy, dispatch, all physical/clock checks and collection. Startup and teardown are measured separately. No physics or acceptance gate is removed from the reported checked-result metric.

Newton uses update_data_interval=0 for this single-SolverMuJoCo task on both CPU and CUDA. MuJoCo retains the evolving internal state; reset explicitly synchronizes the initial joint state, commands and forces still apply each step, and Newton output state still updates each step. This avoids redundant coordinate round trips. It is not a general instruction for coupled solvers that modify each other’s state. Independent CPU models and the replicated GPU model can have different compiled hashes, which are recorded explicitly; this is an application batch cost comparison within Newton’s pinned stack.


This package contains all 36 requested cases, including failed, timed-out or unavailable cases. All 18 configurations per robot must have a terminal result before packaging; accepted cases require one passing warm-up and five passing measurements in every world. Failed cases retain their raw timings for diagnosis but have no accepted median or ratio. Min/max ranges describe the five measured repetitions, not confidence intervals. Timers include reset, controls, 40000 steps and coherent state/solved-contact recording; IK preparation, model setup and compilation are separate. Earlier numerical trials, stress tests and historical full sweeps cannot supply missing repetitions here.

## Hardware and reproducibility

Shared CPU: **AMD Ryzen Threadripper PRO 9985WX 64-Cores**. Selected GPU: **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, **cuda:0**. Visible RAM: **264,948,404,224 bytes**. Versions: mujoco 3.12.0, mujoco-warp 3.12.0, warp-lang 1.17.0, newton 1.6.0, numpy 2.5.3. Complete hardware, driver, CPU affinity, before/after process/load and memory snapshots remain in the raw reports. These snapshots do not establish exclusive access throughout each episode. The second GPU is not combined with the selected device.

Source aggregate: `4b19c4c489b47e62c9d2852530dba3c29fbeb7237c5053906def8e61c8d26ff9`. [Source archive](source.tar.gz) preserves all 64 files of the executed study, including both article stacks, under `sources/articleN/Article_N/`. [Source manifest](source-manifest.json) records their canonical relative paths and both report identities. [Plan](executed-plan.json) preserves the exact launch arguments. The manifest below covers every generated artifact and both compressed and original report hashes. The remote source copy has no Git metadata; file hashes identify what ran.

- SO-101: [JSON.gz](so101/results.json.gz), [CSV](so101/summary.csv), [Markdown](so101/summary.md), [run files](so101/raw-run.tar.gz).
- reBot: [JSON.gz](rebot/results.json.gz), [CSV](rebot/summary.csv), [Markdown](rebot/summary.md), [run files](rebot/raw-run.tar.gz).
- [Audit and failed-world evidence](audit.json), [article section](article-section.md), [structured section](article-section.json), [methods](methodology.md), [artifact manifest](manifest.json).

Gzip decompression restores the original JSON bytes. Run archives retain worker configurations, logs, individual results, initial report and hardware evidence, excluding compiled `.warp-cache` and symlinked cache directories. Ordinary CLI runs retain every world’s verdict and clock/count/capacity receipts; they do not retain the full temporal observation arrays. Separate motion and stress diagnostics retain those larger temporal arrays in the local/remote evidence archive; those multi-gigabyte histories are not bundled here or counted as repetitions in this study. Their one-warm-up/one-measurement checks are not final performance results.

From `Article_3/part3` in the pinned environment, use a new output directory:

```bash
python migration_benchmark.py --scope newton-batch --robot so101 \
  --worlds 1 16 32 64 128 256 512 1024 2048 --cpu-threads 32 --device cuda:0 \
  --nconmax 64 --njmax 128 --repeats 5 --warmups 1 --order-seed 0 \
  --timeout 7200 --max-trajectory-mib 8192 --output-dir .generated/box-full-so101
```

Repeat for `--robot rebot` with another fresh output directory; select your available CUDA device. Revalidate all capacity and task checks on your hardware. The trajectory limit excludes additional model and solver memory. ALOHA/reference replay results belong to their own workload and package; no cross-workload or cross-article ratio is reported here.
