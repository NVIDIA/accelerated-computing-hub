# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **passed**. Robot: **rebot**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| workflow | mujoco | 1 | 1 | passed | 0.2626 / 0.2595–0.2633 | 3.808 |
| workflow | newton_cuda | 1 | 1 | passed | 2.7662 / 2.7319–2.7947 | 0.362 |
| workflow | newton_cpu | 1 | 1 | passed | 1.5519 / 1.4641–1.7168 | 0.644 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

Workflow: optional single-world online controller, simulation and 50 Hz payload observations. Newton reconstructs the model and actuation; these rows are separate workflow measurements and do not enter the identical-model replay speedup calculation.

## Native MuJoCo CPU vs MuJoCo Warp replay

No comparable, fully successful CPU/GPU pair is available; no speedup or crossover is reported.

## Newton CPU vs Newton CUDA

Both paths use Newton's SolverMuJoCo and the online task controller. CPU selects native MuJoCo; CUDA selects MuJoCo Warp. This is a single-world application measurement, including controller and observation overhead; it does not measure batched Newton or solver coupling.

| Robot | Newton CPU (s) | Newton CUDA (s) | CPU / CUDA checked result (s) | Workflow cost ratio (CPU / CUDA) |
|---|---:|---:|---:|---|
| rebot | 1.5519 | 2.7662 | 1.5553 / 2.7692 | 0.561× |

rebot: compiled CPU/CUDA model hashes differ and both are retained in the report. The ratio compares the same Newton application protocol and settings, using each backend's constructed model and state interfaces; it does not establish identical compiled physics.

Times are medians of complete episodes after warm-up; setup is excluded. A reported ratio is CPU time divided by CUDA time, so less than 1× means CUDA was slower. These rows never enter the replay crossover calculation.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
