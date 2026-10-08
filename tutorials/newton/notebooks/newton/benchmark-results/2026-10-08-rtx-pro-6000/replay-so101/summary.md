# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **passed**. Robot: **so101**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mujoco | 256 | 1 | passed | 42.3698 / 42.2359–47.0843 | 6.042 |
| replay | mujoco | 256 | 32 | passed | 1.5937 / 1.5721–1.6242 | 160.632 |
| replay | mjwarp | 1 | 1 | passed | 2.3694 / 2.3554–2.3736 | 0.422 |
| replay | mujoco | 16 | 1 | passed | 2.9837 / 2.9355–3.2156 | 5.362 |
| replay | mujoco | 64 | 1 | passed | 11.3419 / 11.3174–11.5918 | 5.643 |
| replay | mujoco | 16 | 16 | passed | 0.2001 / 0.1952–0.2167 | 79.974 |
| replay | mjwarp | 64 | 1 | passed | 3.0477 / 3.0346–3.0528 | 20.999 |
| replay | mjwarp | 16 | 1 | passed | 2.7977 / 2.7884–2.8050 | 5.719 |
| replay | mujoco | 1 | 1 | passed | 0.2174 / 0.2169–0.2202 | 4.599 |
| replay | mjwarp | 256 | 1 | passed | 3.4102 / 3.4033–3.4181 | 75.070 |
| replay | mujoco | 64 | 32 | passed | 0.4530 / 0.3996–0.4700 | 141.275 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

Workflow: optional single-world online controller, simulation and 50 Hz payload observations. Newton reconstructs the model and actuation; these rows are separate workflow measurements and do not enter the identical-model replay speedup calculation.

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.092× | 0.092× |
| 16 | 16 / 16 | 0.072× | 0.074× |
| 64 | 32 / 32 | 0.149× | 0.156× |
| 256 | 32 / 32 | 0.467× | 0.477× |

No GPU simulation advantage was measured at the tested batch sizes.

No GPU advantage including output transfer and validation was measured at the tested batch sizes.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
