All 36 ALOHA reference cases passed the frozen audit, including one complete warmup and five measured episodes per case. This is native MuJoCo versus MuJoCo Warp replay with state-history recording, including on the Blog 3 software stack. It does not measure the Newton API or validate the two-cube box task.

The shared workstation used an AMD Threadripper PRO 9985WX (up to 32 native C++ rollout threads) and one RTX PRO 6000 Blackwell, cuda:1. The retained hardware inventory identifies the machine; no concurrent-utilization trace was recorded for this reference run.

All times below are median batch seconds. Speedup is CPU/GPU elapsed time at the same environment count; below 1 means the GPU takes longer. The checked columns also include collection and validation. Setup/JIT and artifact writes are excluded.

blog2 software stack

| Environments | CPU simulation (s) | GPU simulation (s) | Simulation speedup | Checked speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.0374 | 0.5296 | 0.071× | 0.072× |
| 16 | 0.0392 | 0.6785 | 0.058× | 0.062× |
| 32 | 0.0397 | 0.7261 | 0.055× | 0.062× |
| 64 | 0.0790 | 0.7729 | 0.102× | 0.118× |
| 128 | 0.1576 | 0.8268 | 0.191× | 0.225× |
| 256 | 0.3339 | 0.8860 | 0.377× | 0.441× |
| 512 | 0.5647 | 0.9472 | 0.596× | 0.664× |
| 1024 | 1.1129 | 1.0568 | 1.053× | 1.049× |
| 2048 | 2.2083 | 1.3071 | 1.689× | 1.506× |

blog3 software stack

| Environments | CPU simulation (s) | GPU simulation (s) | Simulation speedup | Checked speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.0313 | 0.5008 | 0.062× | 0.064× |
| 16 | 0.0322 | 0.6469 | 0.050× | 0.054× |
| 32 | 0.0324 | 0.6811 | 0.048× | 0.056× |
| 64 | 0.0650 | 0.7143 | 0.091× | 0.107× |
| 128 | 0.1286 | 0.7593 | 0.169× | 0.207× |
| 256 | 0.2336 | 0.8117 | 0.288× | 0.360× |
| 512 | 0.5264 | 0.8596 | 0.612× | 0.675× |
| 1024 | 0.9333 | 0.9392 | 0.994× | 1.012× |
| 2048 | 1.8309 | 1.1443 | 1.600× | 1.426× |

The large-batch crossover is workload- and hardware-specific. At 1024 worlds the medians are close to parity; five fixed-tape repeats do not establish a statistical confidence interval. At 2048 worlds the measured GPU advantage is 1.689× / 1.506× for the Blog 2 stack and 1.600× / 1.426× for the Blog 3 stack (simulation / checked). No lower-end GPU, Colab or Brev measurements are represented.

The health audit checks finite trajectories, every integration clock, coarse motion bounds, quaternion norms, final pot/lid heights and GPU capacities. These checks do not prove grasp/contact/hold success. Intermediate trajectories were checked during the run but not retained; raw endpoint files and recorded health summaries remain available. There are zero failed cases in this reference sweep; historical box-task failures are separate evidence.

Files: summary.json contains strict pairing/eligibility and endpoint hashes; paired-results.csv adds timing ranges; episode-timings.csv contains every raw timing; health-summary.json contains every episode health diagnostic; hardware.json and audit-receipt.json retain inventory/supervisor/source provenance.
