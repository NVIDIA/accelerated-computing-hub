# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **passed**. Robot: **so101**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mjwarp | 256 | 1 | passed | 24.0796 / 23.7306–24.1320 | 10.631 |
| replay | mjwarp | 128 | 1 | passed | 22.6108 / 22.5113–22.8191 | 5.661 |
| replay | mujoco | 1 | 1 | passed | 1.2682 / 1.2325–1.5012 | 0.789 |
| replay | mujoco | 2048 | 32 | passed | 90.2232 / 89.9627–90.3237 | 22.699 |
| replay | mujoco | 16 | 16 | passed | 1.4631 / 1.4497–1.4935 | 10.935 |
| replay | mjwarp | 1024 | 1 | passed | 29.2140 / 28.7640–29.5810 | 35.052 |
| replay | mujoco | 256 | 32 | passed | 11.7220 / 11.5771–11.8650 | 21.839 |
| replay | mjwarp | 16 | 1 | passed | 17.9851 / 17.8501–18.2797 | 0.890 |
| replay | mjwarp | 32 | 1 | passed | 19.3022 / 19.2473–19.3505 | 1.658 |
| replay | mjwarp | 2048 | 1 | passed | 34.4835 / 34.4130–34.5124 | 59.391 |
| replay | mujoco | 1024 | 32 | passed | 45.8203 / 45.7117–45.9574 | 22.348 |
| replay | mujoco | 64 | 32 | passed | 2.9472 / 2.9131–2.9570 | 21.715 |
| replay | mjwarp | 64 | 1 | passed | 20.5499 / 20.4591–20.6080 | 3.114 |
| replay | mujoco | 128 | 32 | passed | 5.8159 / 5.7542–5.9495 | 22.009 |
| replay | mujoco | 32 | 32 | passed | 1.4754 / 1.4637–1.5121 | 21.689 |
| replay | mjwarp | 1 | 1 | passed | 15.6355 / 15.5775–15.6584 | 0.064 |
| replay | mjwarp | 512 | 1 | passed | 25.7572 / 25.7149–25.8006 | 19.878 |
| replay | mujoco | 512 | 32 | passed | 23.2236 / 23.0006–23.4386 | 22.047 |

Two-cube box task: each episode runs all 40 seconds (2000 control frames, 40000 physics steps). Episode reset, precomputed controls, a backend forward pass and actual solved-contact observations at 50 Hz are inside the simulation timer. Both backends record the same 61 float32 fields. Native MuJoCo uses persistent independent worker processes; GPU uses batched CUDA graphs. Every cube in every world must pass grasp, lift, carry, detached whole-cube containment and settling gates. Setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. Successful tasks/s measures checked episode throughput, not policy inference, training or rendering.

The full sweep defaults to one CPU pool case per batch; select --cpu-baseline both to add serial cases.

## Native MuJoCo CPU vs MuJoCo Warp replay

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.081× | 0.087× |
| 16 | 16 / 16 | 0.081× | 0.087× |
| 32 | 32 / 32 | 0.076× | 0.083× |
| 64 | 32 / 32 | 0.143× | 0.152× |
| 128 | 32 / 32 | 0.257× | 0.269× |
| 256 | 32 / 32 | 0.487× | 0.503× |
| 512 | 32 / 32 | 0.902× | 0.904× |
| 1024 | 32 / 32 | 1.568× | 1.503× |
| 2048 | 32 / 32 | 2.616× | 2.356× |

First measured batch where GPU simulation was faster: **1024 worlds**. This is an observed point in this sweep, not a universal crossover or an official recommendation.

First measured batch where checked host results were available faster on GPU: **1024 worlds** (setup excluded).

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
