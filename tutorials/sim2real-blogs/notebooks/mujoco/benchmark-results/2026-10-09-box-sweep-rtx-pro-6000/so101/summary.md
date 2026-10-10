# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **incomplete_or_failed**. Robot: **so101**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mjwarp | 256 | 1 | passed | 23.4024 / 23.3051–23.5502 | 10.939 |
| replay | mjwarp | 128 | 1 | passed | 21.4587 / 21.4010–21.5573 | 5.965 |
| replay | mujoco | 1 | 1 | passed | 1.3000 / 1.1089–1.4176 | 0.769 |
| replay | mujoco | 2048 | 32 | passed | 81.7012 / 81.5611–82.0146 | 25.067 |
| replay | mujoco | 16 | 16 | passed | 1.3266 / 1.3036–1.3312 | 12.061 |
| replay | mjwarp | 1024 | 1 | failed | — | — |
| replay | mujoco | 256 | 32 | passed | 10.4544 / 10.4259–10.5357 | 24.487 |
| replay | mjwarp | 16 | 1 | passed | 16.7440 / 16.6728–16.9420 | 0.956 |
| replay | mjwarp | 32 | 1 | passed | 18.1925 / 18.0894–18.2674 | 1.759 |
| replay | mjwarp | 2048 | 1 | failed | — | — |
| replay | mujoco | 1024 | 32 | passed | 41.0932 / 41.0842–41.2119 | 24.919 |
| replay | mujoco | 64 | 32 | passed | 2.6656 / 2.6048–2.6733 | 24.010 |
| replay | mjwarp | 64 | 1 | passed | 19.5688 / 19.5377–19.6651 | 3.271 |
| replay | mujoco | 128 | 32 | passed | 5.2321 / 5.2066–5.2430 | 24.464 |
| replay | mujoco | 32 | 32 | passed | 1.3482 / 1.3233–1.3781 | 23.735 |
| replay | mjwarp | 1 | 1 | passed | 14.6314 / 14.4788–14.6451 | 0.068 |
| replay | mjwarp | 512 | 1 | failed | — | — |
| replay | mujoco | 512 | 32 | passed | 21.1330 / 21.0057–21.2153 | 24.227 |

Two-cube box task: each episode runs all 40 seconds (2000 control frames, 40000 physics steps). Episode reset, precomputed controls, a backend forward pass and actual solved-contact observations at 50 Hz are inside the simulation timer. Both backends record the same 61 float32 fields. Native MuJoCo uses persistent independent worker processes; GPU uses batched CUDA graphs. Every cube in every world must pass grasp, lift, carry, detached whole-cube containment and settling gates. Setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. Successful tasks/s measures checked episode throughput, not policy inference, training or rendering.

The full sweep defaults to one CPU pool case per batch; select --cpu-baseline both to add serial cases.

## Native MuJoCo CPU vs MuJoCo Warp replay

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.089× | 0.094× |
| 16 | 16 / 16 | 0.079× | 0.085× |
| 32 | 32 / 32 | 0.074× | 0.081× |
| 64 | 32 / 32 | 0.136× | 0.145× |
| 128 | 32 / 32 | 0.244× | 0.259× |
| 256 | 32 / 32 | 0.447× | 0.465× |

No GPU simulation advantage was measured at the tested batch sizes.

No GPU advantage including output transfer and validation was measured at the tested batch sizes.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
