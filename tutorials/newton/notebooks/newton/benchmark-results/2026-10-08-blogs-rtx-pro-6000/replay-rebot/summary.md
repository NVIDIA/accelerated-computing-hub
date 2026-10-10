# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **passed**. Robot: **rebot**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mujoco | 16 | 1 | passed | 2.7657 / 2.4858–2.9635 | 5.785 |
| replay | mjwarp | 1 | 1 | passed | 2.9119 / 2.8998–2.9244 | 0.343 |
| replay | mujoco | 1 | 1 | passed | 0.1818 / 0.1579–0.1822 | 5.500 |
| replay | mjwarp | 16 | 1 | passed | 3.5158 / 3.4927–3.5520 | 4.551 |
| replay | mujoco | 16 | 16 | passed | 0.1881 / 0.1816–0.1894 | 85.060 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

Workflow: optional single-world online controller, simulation and 50 Hz payload observations. Newton reconstructs the model and actuation; these rows are separate workflow measurements and do not enter the identical-model replay speedup calculation.

## Native MuJoCo CPU vs MuJoCo Warp replay

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.062× | 0.063× |
| 16 | 16 / 16 | 0.054× | 0.056× |

No GPU simulation advantage was measured at the tested batch sizes.

No GPU advantage including output transfer and validation was measured at the tested batch sizes.

## Newton CPU vs Newton CUDA

Both paths use Newton's SolverMuJoCo and the online task controller. CPU selects native MuJoCo; CUDA selects MuJoCo Warp. This is a single-world application measurement, including controller and observation overhead; it does not measure batched Newton or solver coupling.

No fully successful Newton CPU/CUDA pair is available. Run `--scope workflow` or `--scope both` on a CUDA machine to measure it; failed or unavailable cases remain unpaired.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
