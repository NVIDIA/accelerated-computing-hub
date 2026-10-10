# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **passed**. Robot: **so101**.

| Scope | Backend | Worlds | CPU workers | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| newton-batch | newton_cuda | 256 | 0 | passed | 19.9407 / 19.8684–19.9722 | 12.838 |
| newton-batch | newton_cuda | 128 | 0 | passed | 18.7815 / 18.7527–18.8776 | 6.815 |
| newton-batch | newton_cpu | 1 | 1 | passed | 6.4848 / 6.4707–6.5027 | 0.154 |
| newton-batch | newton_cpu | 2048 | 32 | passed | 482.4144 / 481.9683–483.2738 | 4.245 |
| newton-batch | newton_cpu | 16 | 16 | passed | 7.2303 / 7.2121–7.3061 | 2.213 |
| newton-batch | newton_cuda | 1024 | 0 | passed | 23.8462 / 23.6913–23.9272 | 42.942 |
| newton-batch | newton_cpu | 256 | 32 | passed | 60.7351 / 60.4972–61.7740 | 4.215 |
| newton-batch | newton_cuda | 16 | 0 | passed | 16.7782 / 16.6881–16.8911 | 0.954 |
| newton-batch | newton_cuda | 32 | 0 | passed | 17.2842 / 17.1650–17.3169 | 1.851 |
| newton-batch | newton_cuda | 2048 | 0 | passed | 28.3054 / 28.2730–28.3469 | 72.354 |
| newton-batch | newton_cpu | 1024 | 32 | passed | 242.4549 / 242.3513–243.1436 | 4.223 |
| newton-batch | newton_cpu | 64 | 32 | passed | 15.2200 / 15.1481–15.3146 | 4.205 |
| newton-batch | newton_cuda | 64 | 0 | passed | 17.8851 / 17.7840–17.9217 | 3.578 |
| newton-batch | newton_cpu | 128 | 32 | passed | 30.4011 / 30.3645–30.6490 | 4.210 |
| newton-batch | newton_cpu | 32 | 32 | passed | 7.6543 / 7.6015–7.8387 | 4.181 |
| newton-batch | newton_cuda | 1 | 0 | passed | 14.7928 / 14.6127–14.9309 | 0.068 |
| newton-batch | newton_cuda | 512 | 0 | passed | 21.6250 / 21.5319–21.6779 | 23.676 |
| newton-batch | newton_cpu | 512 | 32 | passed | 122.6874 / 122.1557–124.3836 | 4.173 |

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
| 1 | 1 | 6.4848 | 14.7928 | 0.438× | 0.442× |
| 16 | 16 | 7.2303 | 16.7782 | 0.431× | 0.434× |
| 32 | 32 | 7.6543 | 17.2842 | 0.443× | 0.447× |
| 64 | 32 | 15.2200 | 17.8851 | 0.851× | 0.853× |
| 128 | 32 | 30.4011 | 18.7815 | 1.619× | 1.604× |
| 256 | 32 | 60.7351 | 19.9407 | 3.046× | 2.962× |
| 512 | 32 | 122.6874 | 21.6250 | 5.673× | 5.341× |
| 1024 | 32 | 242.4549 | 23.8462 | 10.167× | 9.055× |
| 2048 | 32 | 482.4144 | 28.3054 | 17.043× | 13.998× |

Ratios divide CPU time by CUDA time: above 1× means CUDA finished the measured batch faster. Each pair requires equal environment counts, source/protocol identity and all worlds passing. Backend model hashes remain recorded; the ratio does not establish identical compiled physics. These rows are separate from the one-world online-controller workflow and native MuJoCo replay.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
