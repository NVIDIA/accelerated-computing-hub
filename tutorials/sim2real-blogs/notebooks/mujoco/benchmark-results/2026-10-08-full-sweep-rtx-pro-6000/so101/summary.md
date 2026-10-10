# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **incomplete_or_failed**. Robot: **so101**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mjwarp | 256 | 1 | failed | — | — |
| replay | mjwarp | 128 | 1 | passed | 4.0895 / 4.0545–4.1315 | 31.299 |
| replay | mujoco | 1 | 1 | passed | 0.2170 / 0.2142–0.2201 | 4.609 |
| replay | mujoco | 2048 | 32 | passed | 14.5960 / 14.5729–14.9211 | 140.312 |
| replay | mujoco | 16 | 16 | passed | 0.2276 / 0.2242–0.2305 | 70.293 |
| replay | mjwarp | 1024 | 1 | passed | 5.6639 / 5.6520–5.6786 | 180.793 |
| replay | mujoco | 256 | 32 | passed | 1.8164 / 1.7941–1.8341 | 140.935 |
| replay | mjwarp | 16 | 1 | passed | 3.2366 / 3.2226–3.2848 | 4.943 |
| replay | mjwarp | 32 | 1 | passed | 3.5349 / 3.5105–3.5555 | 9.053 |
| replay | mjwarp | 2048 | 1 | passed | 7.0647 / 7.0201–7.0757 | 289.893 |
| replay | mujoco | 1024 | 32 | passed | 7.4726 / 7.3998–7.6511 | 137.034 |
| replay | mujoco | 64 | 32 | passed | 0.4798 / 0.4688–0.5269 | 133.388 |
| replay | mjwarp | 64 | 1 | passed | 3.7822 / 3.7542–3.8036 | 16.921 |
| replay | mujoco | 128 | 32 | passed | 0.9617 / 0.9494–1.0305 | 133.093 |
| replay | mujoco | 32 | 32 | passed | 0.2296 / 0.2261–0.2659 | 139.361 |
| replay | mjwarp | 1 | 1 | passed | 2.6458 / 2.6445–2.6711 | 0.378 |
| replay | mjwarp | 512 | 1 | passed | 4.9391 / 4.8957–4.9533 | 103.663 |
| replay | mujoco | 512 | 32 | passed | 3.6730 / 3.6136–3.7057 | 139.395 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

The CPU baseline uses the configured serial or persistent worker pool. The full sweep defaults to one CPU pool case per batch; select --cpu-baseline both to add serial cases.

## Native MuJoCo CPU vs MuJoCo Warp replay

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.082× | 0.082× |
| 16 | 16 / 16 | 0.070× | 0.072× |
| 32 | 32 / 32 | 0.065× | 0.069× |
| 64 | 32 / 32 | 0.127× | 0.133× |
| 128 | 32 / 32 | 0.235× | 0.248× |
| 512 | 32 / 32 | 0.744× | 0.742× |
| 1024 | 32 / 32 | 1.319× | 1.259× |
| 2048 | 32 / 32 | 2.066× | 1.880× |

First measured batch where GPU simulation was faster: **1024 worlds**. This is an observed point in this sweep, not a universal crossover or an official recommendation.

First measured batch where checked host results were available faster on GPU: **1024 worlds** (setup excluded).

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
