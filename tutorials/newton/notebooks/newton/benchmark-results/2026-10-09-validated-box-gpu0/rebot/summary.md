# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **incomplete_or_failed**. Robot: **rebot**.

| Scope | Backend | Worlds | CPU workers | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| newton-batch | newton_cuda | 256 | 0 | passed | 20.6777 / 20.5243–20.8301 | 12.380 |
| newton-batch | newton_cuda | 128 | 0 | passed | 19.3498 / 19.3273–19.4506 | 6.615 |
| newton-batch | newton_cpu | 1 | 1 | passed | 6.3294 / 6.3014–7.4459 | 0.158 |
| newton-batch | newton_cpu | 2048 | 32 | passed | 475.6768 / 474.5915–476.0257 | 4.305 |
| newton-batch | newton_cpu | 16 | 16 | passed | 7.4193 / 7.4069–7.4601 | 2.157 |
| newton-batch | newton_cuda | 1024 | 0 | failed | — | — |
| newton-batch | newton_cpu | 256 | 32 | passed | 59.7683 / 59.6576–59.9986 | 4.283 |
| newton-batch | newton_cuda | 16 | 0 | passed | 17.1097 / 17.0128–17.3197 | 0.935 |
| newton-batch | newton_cuda | 32 | 0 | passed | 17.6607 / 17.6020–17.7211 | 1.812 |
| newton-batch | newton_cuda | 2048 | 0 | passed | 30.0213 / 29.8752–30.0516 | 68.218 |
| newton-batch | newton_cpu | 1024 | 32 | passed | 238.9082 / 238.4800–239.9356 | 4.286 |
| newton-batch | newton_cpu | 64 | 32 | passed | 15.1986 / 15.0038–15.6481 | 4.211 |
| newton-batch | newton_cuda | 64 | 0 | passed | 18.6563 / 18.5588–18.7945 | 3.430 |
| newton-batch | newton_cpu | 128 | 32 | passed | 29.9407 / 29.8560–30.0051 | 4.275 |
| newton-batch | newton_cpu | 32 | 32 | passed | 7.4869 / 7.4665–7.5105 | 4.274 |
| newton-batch | newton_cuda | 1 | 0 | passed | 15.5382 / 15.4346–15.6747 | 0.064 |
| newton-batch | newton_cuda | 512 | 0 | passed | 22.3135 / 22.2304–22.4301 | 22.946 |
| newton-batch | newton_cpu | 512 | 32 | passed | 119.8159 / 119.7758–120.7203 | 4.273 |

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
| 1 | 1 | 6.3294 | 15.5382 | 0.407× | 0.411× |
| 16 | 16 | 7.4193 | 17.1097 | 0.434× | 0.437× |
| 32 | 32 | 7.4869 | 17.6607 | 0.424× | 0.428× |
| 64 | 32 | 15.1986 | 18.6563 | 0.815× | 0.817× |
| 128 | 32 | 29.9407 | 19.3498 | 1.547× | 1.532× |
| 256 | 32 | 59.7683 | 20.6777 | 2.890× | 2.807× |
| 512 | 32 | 119.8159 | 22.3135 | 5.370× | 5.062× |
| 2048 | 32 | 475.6768 | 30.0213 | 15.845× | 13.190× |

Ratios divide CPU time by CUDA time: above 1× means CUDA finished the measured batch faster. Each pair requires equal environment counts, source/protocol identity and all worlds passing. Backend model hashes remain recorded; the ratio does not establish identical compiled physics. These rows are separate from the one-world online-controller workflow and native MuJoCo replay.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
