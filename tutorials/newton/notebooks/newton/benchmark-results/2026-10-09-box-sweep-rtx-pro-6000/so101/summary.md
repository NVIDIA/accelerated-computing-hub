# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **incomplete_or_failed**. Robot: **so101**.

| Scope | Backend | Worlds | CPU workers | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| newton-batch | newton_cuda | 256 | 0 | failed | — | — |
| newton-batch | newton_cuda | 128 | 0 | failed | — | — |
| newton-batch | newton_cpu | 1 | 1 | passed | 7.6646 / 7.6484–7.7213 | 0.130 |
| newton-batch | newton_cpu | 2048 | 32 | passed | 568.6423 / 567.7765–569.7236 | 3.602 |
| newton-batch | newton_cpu | 16 | 16 | passed | 8.8893 / 8.8397–8.9169 | 1.800 |
| newton-batch | newton_cuda | 1024 | 0 | failed | — | — |
| newton-batch | newton_cpu | 256 | 32 | passed | 71.8295 / 71.7360–72.3445 | 3.564 |
| newton-batch | newton_cuda | 16 | 0 | passed | 16.0423 / 16.0147–16.0942 | 0.997 |
| newton-batch | newton_cuda | 32 | 0 | passed | 16.5255 / 16.4797–16.6136 | 1.936 |
| newton-batch | newton_cuda | 2048 | 0 | failed | — | — |
| newton-batch | newton_cpu | 1024 | 32 | passed | 286.5794 / 285.5843–287.7911 | 3.573 |
| newton-batch | newton_cpu | 64 | 32 | passed | 18.0403 / 17.9981–18.1285 | 3.548 |
| newton-batch | newton_cuda | 64 | 0 | passed | 17.2860 / 17.2387–17.4035 | 3.702 |
| newton-batch | newton_cpu | 128 | 32 | passed | 35.8996 / 35.6904–36.2200 | 3.565 |
| newton-batch | newton_cpu | 32 | 32 | passed | 8.9924 / 8.9494–9.0298 | 3.559 |
| newton-batch | newton_cuda | 1 | 0 | passed | 14.0876 / 13.9078–14.2377 | 0.071 |
| newton-batch | newton_cuda | 512 | 0 | failed | — | — |
| newton-batch | newton_cpu | 512 | 32 | passed | 143.0035 / 142.6825–144.7633 | 3.580 |

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
| 1 | 1 | 7.6646 | 14.0876 | 0.544× | 0.547× |
| 16 | 16 | 8.8893 | 16.0423 | 0.554× | 0.557× |
| 32 | 32 | 8.9924 | 16.5255 | 0.544× | 0.548× |
| 64 | 32 | 18.0403 | 17.2860 | 1.044× | 1.040× |

Ratios divide CPU time by CUDA time: above 1× means CUDA finished the measured batch faster. Each pair requires equal environment counts, source/protocol identity and all worlds passing. Backend model hashes remain recorded; the ratio does not establish identical compiled physics. These rows are separate from the one-world online-controller workflow and native MuJoCo replay.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
