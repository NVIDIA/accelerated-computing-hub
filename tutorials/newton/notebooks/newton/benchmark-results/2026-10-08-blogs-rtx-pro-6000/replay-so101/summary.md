# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **passed**. Robot: **so101**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mujoco | 256 | 1 | passed | 49.0243 / 48.1194–49.6419 | 5.222 |
| replay | mujoco | 256 | 32 | passed | 1.5785 / 1.5696–1.5880 | 162.175 |
| replay | mjwarp | 1 | 1 | passed | 2.3671 / 2.3619–2.3804 | 0.422 |
| replay | mujoco | 16 | 1 | passed | 3.4069 / 3.1975–3.4912 | 4.696 |
| replay | mujoco | 64 | 1 | passed | 12.3154 / 12.0933–13.0307 | 5.197 |
| replay | mujoco | 16 | 16 | passed | 0.2250 / 0.2184–0.2289 | 71.106 |
| replay | mjwarp | 64 | 1 | passed | 3.0428 / 3.0365–3.0546 | 21.033 |
| replay | mjwarp | 16 | 1 | passed | 2.8058 / 2.7848–2.8270 | 5.703 |
| replay | mujoco | 1 | 1 | passed | 0.2195 / 0.1820–0.2254 | 4.555 |
| replay | mjwarp | 256 | 1 | passed | 3.4175 / 3.4036–3.4228 | 74.909 |
| replay | mujoco | 64 | 32 | passed | 0.4320 / 0.4021–0.4573 | 148.135 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

Workflow: optional single-world online controller, simulation and 50 Hz payload observations. Newton reconstructs the model and actuation; these rows are separate workflow measurements and do not enter the identical-model replay speedup calculation.

## Native MuJoCo CPU vs MuJoCo Warp replay

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.093× | 0.093× |
| 16 | 16 / 16 | 0.080× | 0.083× |
| 64 | 32 / 32 | 0.142× | 0.150× |
| 256 | 32 / 32 | 0.462× | 0.472× |

No GPU simulation advantage was measured at the tested batch sizes.

No GPU advantage including output transfer and validation was measured at the tested batch sizes.

## Newton CPU vs Newton CUDA

Both paths use Newton's SolverMuJoCo and the online task controller. CPU selects native MuJoCo; CUDA selects MuJoCo Warp. This is a single-world application measurement, including controller and observation overhead; it does not measure batched Newton or solver coupling.

No fully successful Newton CPU/CUDA pair is available. Run `--scope workflow` or `--scope both` on a CUDA machine to measure it; failed or unavailable cases remain unpaired.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
