# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **incomplete_or_failed**. Robot: **rebot**.

| Scope | Backend | Worlds | CPU workers | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| newton-batch | newton_cuda | 256 | 0 | failed | — | — |
| newton-batch | newton_cuda | 128 | 0 | failed | — | — |
| newton-batch | newton_cpu | 1 | 1 | passed | 7.6064 / 7.5538–7.9960 | 0.131 |
| newton-batch | newton_cpu | 2048 | 32 | passed | 562.0810 / 561.8343–562.4356 | 3.644 |
| newton-batch | newton_cpu | 16 | 16 | passed | 8.8661 / 8.8173–8.8952 | 1.805 |
| newton-batch | newton_cuda | 1024 | 0 | failed | — | — |
| newton-batch | newton_cpu | 256 | 32 | passed | 70.9853 / 70.6217–71.2983 | 3.606 |
| newton-batch | newton_cuda | 16 | 0 | passed | 17.4278 / 17.3981–17.5367 | 0.918 |
| newton-batch | newton_cuda | 32 | 0 | passed | 18.3622 / 18.2601–18.5001 | 1.743 |
| newton-batch | newton_cuda | 2048 | 0 | failed | — | — |
| newton-batch | newton_cpu | 1024 | 32 | passed | 282.9764 / 282.0601–283.8140 | 3.619 |
| newton-batch | newton_cpu | 64 | 32 | passed | 17.8876 / 17.7563–18.0949 | 3.578 |
| newton-batch | newton_cuda | 64 | 0 | passed | 19.3339 / 19.1933–19.8014 | 3.310 |
| newton-batch | newton_cpu | 128 | 32 | passed | 35.5810 / 35.5336–35.6745 | 3.597 |
| newton-batch | newton_cpu | 32 | 32 | passed | 8.9584 / 8.9349–9.0665 | 3.572 |
| newton-batch | newton_cuda | 1 | 0 | passed | 15.4567 / 15.3469–15.6557 | 0.065 |
| newton-batch | newton_cuda | 512 | 0 | failed | — | — |
| newton-batch | newton_cpu | 512 | 32 | passed | 141.6576 / 141.4189–141.7078 | 3.614 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

Workflow: optional single-world online controller, simulation and 50 Hz payload observations. Newton reconstructs the model and actuation; these rows are separate workflow measurements and do not enter the identical-model replay speedup calculation.

## Native MuJoCo CPU vs MuJoCo Warp replay

No comparable, fully successful CPU/GPU pair is available; no speedup or crossover is reported.

## Newton CPU vs Newton CUDA

Both paths use Newton's SolverMuJoCo and the online task controller. CPU selects native MuJoCo; CUDA selects MuJoCo Warp. This is a single-world application measurement, including controller and observation overhead; it does not measure batched Newton or solver coupling.

No fully successful Newton CPU/CUDA pair is available. Run `--scope workflow` or `--scope both` on a CUDA machine to measure it; failed or unavailable cases remain unpaired.

## Newton CPU vs Newton CUDA across environment counts

CPU runs independent Newton SolverMuJoCo instances in persistent worker processes; CUDA advances a replicated Newton model. Both pick each cube and release it inside the receiving box over 40 simulated seconds in every environment. Timed batches include reset, stepping, forward evaluations and live geometry/solved-contact recording; setup, warm-up, final transfer and task validation are separately recorded.

| Environments | CPU processes | Newton CPU (s) | Newton CUDA (s) | CPU/CUDA batch cost ratio | Checked-result ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.6064 | 15.4567 | 0.492× | 0.495× |
| 16 | 16 | 8.8661 | 17.4278 | 0.509× | 0.511× |
| 32 | 32 | 8.9584 | 18.3622 | 0.488× | 0.492× |
| 64 | 32 | 17.8876 | 19.3339 | 0.925× | 0.925× |

Ratios divide CPU time by CUDA time: above 1× means CUDA finished the measured batch faster. Each pair requires equal environment counts, source/protocol identity and all worlds passing. Backend model hashes remain recorded; the ratio does not establish identical compiled physics. These rows are separate from the one-world online-controller workflow and native MuJoCo replay.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
