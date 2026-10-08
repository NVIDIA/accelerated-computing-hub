# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **passed**. Robot: **rebot**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mujoco | 16 | 1 | passed | 2.4531 / 2.3942–2.7018 | 6.522 |
| replay | mjwarp | 1 | 1 | passed | 2.9153 / 2.9009–2.9204 | 0.343 |
| replay | mujoco | 1 | 1 | passed | 0.1892 / 0.1803–0.1904 | 5.286 |
| replay | mjwarp | 16 | 1 | passed | 3.5142 / 3.4951–3.5224 | 4.553 |
| replay | mujoco | 16 | 16 | passed | 0.1900 / 0.1880–0.1911 | 84.192 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

Workflow: optional single-world online controller, simulation and 50 Hz payload observations. Newton reconstructs the model and actuation; these rows are separate workflow measurements and do not enter the identical-model replay speedup calculation.

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.065× | 0.065× |
| 16 | 16 / 16 | 0.054× | 0.056× |

No GPU simulation advantage was measured at the tested batch sizes.

No GPU advantage including output transfer and validation was measured at the tested batch sizes.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
