# CPU/GPU reference: ALOHA control replay

This self-contained companion compares native MuJoCo on CPU with MuJoCo Warp on one GPU. It uses an adapted upstream ALOHA pot/lid replay, with the same environment count, initial state, control tape, integration count and effective solver settings for each CPU/GPU pair. **It is a community measurement, separate from the blogs’ two-cube receiving-box task. It does not time the Newton API**, including when run with the Blog 3 package versions.

The retained workstation sweep covers **1, 16, 32, 64, 128, 256, 512, 1024 and 2048 environments** for both software stacks. All 36 cases passed their frozen eligibility checks: one complete warmup and five measured episodes, each containing 1001 integrations at 2 ms. No rendering, online inverse kinematics or learning is timed.

| Software stack | At 2048 environments: simulation speedup | Including collection and checks |
|---|---:|---:|
| MuJoCo 3.8.0 / MuJoCo Warp 3.8.0.3 / Warp 1.15.0 | 1.689× | 1.506× |
| MuJoCo / MuJoCo Warp 3.12.0 / Warp 1.17.0 | 1.600× | 1.426× |

Speedup is CPU elapsed time divided by GPU elapsed time at the **same** batch size. The GPU was slower through 512 environments in this run; the medians were near parity at 1024. Five repeats of a fixed tape do not establish a confidence interval or randomized-task robustness. The full tables include every requested size and per-episode timings.

The shared workstation used an AMD Ryzen Threadripper PRO 9985WX, up to 32 native C++ rollout threads, and one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`cuda:1`). [Hardware evidence](audit/hardware.json) links the retained inventory. No utilization trace was recorded during this reference sweep, and exclusive use of the machine is not claimed. Developer GPUs, Colab and Brev remain unvalidated.

## Inspect the results

- [Results and interpretation](audit/RESULTS.md), [paired CSV](audit/paired-results.csv), and [all episode timings](audit/episode-timings.csv).
- [Strict eligibility and pairing](audit/summary.json), [health summaries](audit/health-summary.json), and [independent audit receipt](audit/audit-receipt.json).
- [Raw JSON reports and endpoint NPZ files](evidence/results/) and [all 36 launch logs](evidence/logs/).
- Figures for [the Blog 2 stack](audit/figures-blog2/blog2-reference.png) ([SVG](audit/figures-blog2/blog2-reference.svg), [PDF](audit/figures-blog2/blog2-reference.pdf)) and [the Blog 3 stack](audit/figures-blog3/blog3-reference.png) ([SVG](audit/figures-blog3/blog3-reference.svg), [PDF](audit/figures-blog3/blog3-reference.pdf)).

The simulation timer includes reset, replay, integration, complete state-history recording and dispatch/completion. The second timer adds collection and numerical/physical checks. Setup/JIT and writing artifacts to disk are excluded. CPU history is float64; GPU history is float32. Coarse health checks cover clocks, finite states, quaternion norms, motion bounds, capacities and final pot/lid heights. They do not prove contact-based grasp/hold success. Intermediate histories were checked in memory; only endpoint states and health summaries were retained. No failed reference case is hidden. Historical box-task failures are separate evidence.

## Reproduce

Follow [RUNNING.md](RUNNING.md) for pinned dependencies, complete sweeps and bounded Colab/Brev starting instructions. It uses only paths relative to this folder or an explicit asset cache. Use fresh output directories to preserve previous runs. The asset fetcher downloads immutable, hashed upstream files; **meshes and fetched assets are not bundled here**. Read [ASSET_ORIGINS.md](ASSET_ORIGINS.md) before redistributing assets: upstream licenses differ, and the specific pot/lid terms remain unresolved.

To independently collect the retained evidence from this folder after installing the pinned requirements:

```bash
python collect_reference_sweep.py --results-root evidence/results --device cuda:1 --workers 32 > reviewed-summary.json
python make_figures.py --summary reviewed-summary.json --stack blog2 --output reviewed-figures-blog2
python make_figures.py --summary reviewed-summary.json --stack blog3 --output reviewed-figures-blog3
```

The supplied recipe code is copied byte-for-byte from the approved preparation; [copied source identities](provenance/copied-source-identity.json) and `RELEASE_MANIFEST.json` record the hashes. Original audit reports/scripts and workstation plans retain their original absolute paths as historical provenance. They are not launch instructions for a new machine: use the portable recipe above and [RUNNING.md](RUNNING.md). No cross-article repository dependency is required.
