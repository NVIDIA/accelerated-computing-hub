# Article 2: full CPU/GPU sweep — October 8, 2026

Native MuJoCo CPU versus MuJoCo Warp, measured at **1, 16, 32, 64, 128, 256, 512, 1024 and 2048 worlds** for both robots. These are community task measurements on shared hardware, not official product benchmarks or a minimum hardware specification. All **36 requested cases** are retained: **33 passed**, **3 failed or incomplete**. A failed size has no accepted timing median or CPU/GPU ratio.

## Validated timings

![CPU and GPU batch times for SO-101 and reBot, with gaps at failed GPU configurations.](figures/fig_cpu_gpu_scaling.png)

Download the [vector PDF](figures/fig_cpu_gpu_scaling.pdf) or [SVG](figures/fig_cpu_gpu_scaling.svg). The [rendering script](figures/gen_fig_cpu_gpu_scaling.py) reads the compressed reports in this directory and requires Matplotlib; it does not change the physics environment. [Plotted values](figures/fig_cpu_gpu_scaling.data.json) and a [full caption](figures/fig_cpu_gpu_scaling.caption.txt) accompany the figure.

Values are median seconds for five complete 12-second episodes after one full warm-up. CPU/GPU is CPU time divided by GPU time; below 1× means the GPU took longer. The CPU pool uses up to 32 workers, capped by world count. This sweep measures that pool baseline; it does not supply a separate serial-CPU comparison at every size.

| Worlds | CPU workers | SO-101 CPU (s) | SO-101 GPU (s) | CPU/GPU | reBot CPU (s) | reBot GPU (s) | CPU/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.2170 | 2.6458 | 0.082× | 0.1756 | 3.2123 | 0.055× |
| 16 | 16 | 0.2276 | 3.2366 | 0.070× | 0.1958 | 4.0329 | 0.049× |
| 32 | 32 | 0.2296 | 3.5349 | 0.065× | 0.2139 | 4.2905 | 0.050× |
| 64 | 32 | 0.4798 | 3.7822 | 0.127× | 0.3749 | 4.5911 | 0.082× |
| 128 | 32 | 0.9617 | 4.0895 | 0.235× | 0.7519 | 4.9066 | 0.153× |
| 256 | 32 | 1.8164 | **failed** | — | 1.5419 | 5.3292 | 0.289× |
| 512 | 32 | 3.6730 | 4.9391 | 0.744× | 2.9624 | **failed** | — |
| 1024 | 32 | 7.4726 | 5.6639 | 1.319× | 5.8830 | 6.5201 | 0.902× |
| 2048 | 32 | 14.5960 | 7.0647 | 2.066× | 11.8417 | **failed** | — |

SO-101: first accepted tested size with faster GPU simulation **1024 worlds**; with checked host results **1024 worlds**; reBot: first accepted tested size with faster GPU simulation **none**; with checked host results **none**. These are observations at the tested sizes, not universal crossover thresholds; failed cases are excluded.

CPU uses a persistent native MuJoCo rollout worker pool. GPU replays the same compiled model, initial state and float32-quantized controls. Full physics state is recorded after each of the 6,000 steps: float64 in CPU RAM and float32 in GPU memory. Both paths use 100 solver iterations, 50 line-search iterations and `impratio=100`. Timers include episode reset, stepping and observation recording, with CUDA synchronization or CPU worker dispatch/completion. Setup, warm-up, host output collection and validation are recorded separately. The raw reports also give timings through checked host results. Article 2 and Article 3 use different dependency versions and observation policies; their tables do not establish a cross-article speedup. See [methods and acceptance criteria](../../BENCHMARK.md).

## Task outcomes

Every accepted case passed in every world during all six episodes: complete duration, finite state, airborne pickup/continuous carry, stacking on the table, settling and capacity checks. Settling requires RMS point-speed bound ≤0.05 m/s, position diameter ≤5 mm and orientation diameter ≤3° over the final second. Failed cases stop at the first rejected episode; earlier successful repetitions do not make that case eligible.

- SO-101, mjwarp, 256 worlds: orientation diameter >3° (1 world-episode). Recorded episode counts: 1 warm-up, 2 measured; excluded from timing ratios.
- reBot, mjwarp, 512 worlds: orientation diameter >3° (1 world-episode). Recorded episode counts: 1 warm-up, 3 measured; excluded from timing ratios.
- reBot, mjwarp, 2048 worlds: position diameter >5 mm (4 world-episodes); RMS point-speed bound >0.05 m/s (1 world-episode). Recorded episode counts: 1 warm-up, 1 measured; excluded from timing ratios.

The [audit](audit.json) retains failed-world values and verifies count coverage, source/model/control signatures, every accepted warm-up/repetition, and exclusion from ratios. Its episode counts include all recorded attempts, including failed episodes; omitted later repetitions are not imputed.

## Hardware and provenance

CPU: **AMD Ryzen Threadripper PRO 9985WX 64-Cores**, 64 physical cores / 128 logical CPUs; RAM **264,948,391,936 bytes** (246.75 GiB visible). Selected GPU: **cuda:1, NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, 97,887 MiB VRAM, driver **615.71.09**. The second GPU is not combined with the selected device.

Linux 7.0.0-38-generic x86_64; Python 3.12.13; mujoco 3.8.0, mujoco-warp 3.8.0.3, warp-lang 1.15.0, numpy 2.5.3. Other workloads and resident GPU allocations remained on the machine. Before/after utilization, memory and process snapshots document shared use; they do not prove exclusive hardware access throughout each episode. No lower-end GPU, Colab runtime or live Brev deployment was validated by these runs.

Both reports have source aggregate `7ec61addda2705f773c2e7bd4f92be77703a294cf0601186c3d90fedf4eb116a`. All recorded source-file hashes were checked against this article's source tree before packaging. The [executed source archive](source.tar.gz) retains canonical `Article_2/…` paths; every archived file was byte-verified against the checkout, including dependency files. The remote snapshot omitted `.git`, so source hashes identify the executed files. [Manifest](manifest.json) records source, archive-member and compressed/uncompressed report digests; gzip decompression restores the exact original JSON bytes.

- SO-101: [raw JSON.gz](so101/results.json.gz), [CSV](so101/summary.csv), [report](so101/summary.md), [complete run files](so101/raw-run.tar.gz).
- reBot: [raw JSON.gz](rebot/results.json.gz), [CSV](rebot/summary.csv), [report](rebot/summary.md), [complete run files](rebot/raw-run.tar.gz).

Run archives preserve preflight, every worker configuration, individual JSON reports and logs, including failures. Compiled `.warp-cache` files are excluded.

From `Article_2/part2` in its locked environment, reproduce each robot in a fresh output directory; choose your available CUDA device:

```bash
python migration_benchmark.py --cpu-baseline pool --robot so101 \
  --worlds 1 16 32 64 128 256 512 1024 2048 --cpu-threads 32 --device cuda:1 \
  --repeats 5 --warmups 1 --max-trajectory-mib 8192 --output-dir .generated/full-so101
```

Repeat with `--robot rebot` and `.generated/full-rebot`. The 8 GiB trajectory limit excludes additional model/solver memory. Preserve failed rows and revalidate on the developer hardware of interest before drawing broader conclusions.
