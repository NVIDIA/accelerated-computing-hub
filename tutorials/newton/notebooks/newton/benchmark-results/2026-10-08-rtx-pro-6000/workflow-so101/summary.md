# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **passed**. Robot: **so101**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| workflow | mujoco | 1 | 1 | passed | 0.2817 / 0.2798–0.2830 | 3.550 |
| workflow | newton_cuda | 1 | 1 | passed | 2.3273 / 2.3227–2.3289 | 0.430 |
| workflow | newton_cpu | 1 | 1 | passed | 1.5878 / 1.4916–1.6758 | 0.630 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

Workflow: optional single-world online controller, simulation and 50 Hz payload observations. Newton reconstructs the model and actuation; these rows are separate workflow measurements and do not enter the identical-model replay speedup calculation.

No comparable, fully successful CPU/GPU pair is available; no speedup or crossover is reported.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
