# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **incomplete_or_failed**. Robot: **rebot**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mjwarp | 256 | 1 | passed | 27.3368 / 27.2724–27.4825 | 9.365 |
| replay | mjwarp | 128 | 1 | passed | 25.4998 / 25.4501–25.5525 | 5.020 |
| replay | mujoco | 1 | 1 | passed | 1.1887 / 1.1230–1.4610 | 0.841 |
| replay | mujoco | 2048 | 32 | passed | 83.1366 / 83.0579–83.2720 | 24.634 |
| replay | mujoco | 16 | 16 | passed | 1.3369 / 1.3278–1.3616 | 11.968 |
| replay | mjwarp | 1024 | 1 | passed | 32.0479 / 31.9359–32.1427 | 31.952 |
| replay | mujoco | 256 | 32 | passed | 10.6702 / 10.6645–10.7925 | 23.992 |
| replay | mjwarp | 16 | 1 | passed | 21.5820 / 21.4901–21.6202 | 0.741 |
| replay | mjwarp | 32 | 1 | passed | 23.0548 / 23.0073–23.1425 | 1.388 |
| replay | mjwarp | 2048 | 1 | passed | 38.1011 / 38.0419–38.1042 | 53.752 |
| replay | mujoco | 1024 | 32 | passed | 42.0190 / 41.8647–42.0623 | 24.370 |
| replay | mujoco | 64 | 32 | passed | 2.6919 / 2.6449–2.7007 | 23.775 |
| replay | mjwarp | 64 | 1 | passed | 24.0782 / 24.0240–24.1051 | 2.658 |
| replay | mujoco | 128 | 32 | passed | 5.3933 / 5.3892–5.4508 | 23.733 |
| replay | mujoco | 32 | 32 | passed | 1.3536 / 1.3455–1.4076 | 23.640 |
| replay | mjwarp | 1 | 1 | passed | 19.4162 / 19.3750–19.5156 | 0.052 |
| replay | mjwarp | 512 | 1 | failed | — | — |
| replay | mujoco | 512 | 32 | passed | 21.1583 / 21.0350–21.2808 | 24.199 |

Two-cube box task: each episode runs all 40 seconds (2000 control frames, 40000 physics steps). Episode reset, precomputed controls, a backend forward pass and actual solved-contact observations at 50 Hz are inside the simulation timer. Both backends record the same 61 float32 fields. Native MuJoCo uses persistent independent worker processes; GPU uses batched CUDA graphs. Every cube in every world must pass grasp, lift, carry, detached whole-cube containment and settling gates. Setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. Successful tasks/s measures checked episode throughput, not policy inference, training or rendering.

The full sweep defaults to one CPU pool case per batch; select --cpu-baseline both to add serial cases.

## Native MuJoCo CPU vs MuJoCo Warp replay

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.061× | 0.065× |
| 16 | 16 / 16 | 0.062× | 0.066× |
| 32 | 32 / 32 | 0.059× | 0.064× |
| 64 | 32 / 32 | 0.112× | 0.120× |
| 128 | 32 / 32 | 0.212× | 0.224× |
| 256 | 32 / 32 | 0.390× | 0.406× |
| 1024 | 32 / 32 | 1.311× | 1.281× |
| 2048 | 32 / 32 | 2.182× | 2.003× |

First measured batch where GPU simulation was faster: **1024 worlds**. This is an observed point in this sweep, not a universal crossover or an official recommendation.

First measured batch where checked host results were available faster on GPU: **1024 worlds** (setup excluded).

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
