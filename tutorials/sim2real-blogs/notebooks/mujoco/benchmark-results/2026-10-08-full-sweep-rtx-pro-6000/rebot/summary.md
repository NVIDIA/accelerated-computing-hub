# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **incomplete_or_failed**. Robot: **rebot**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mjwarp | 256 | 1 | passed | 5.3292 / 5.3167–5.3385 | 48.037 |
| replay | mjwarp | 128 | 1 | passed | 4.9066 / 4.8922–4.9163 | 26.087 |
| replay | mujoco | 1 | 1 | passed | 0.1756 / 0.1700–0.1763 | 5.695 |
| replay | mujoco | 2048 | 32 | passed | 11.8417 / 11.6714–11.9621 | 172.948 |
| replay | mujoco | 16 | 16 | passed | 0.1958 / 0.1880–0.2110 | 81.737 |
| replay | mjwarp | 1024 | 1 | passed | 6.5201 / 6.5152–6.5264 | 157.052 |
| replay | mujoco | 256 | 32 | passed | 1.5419 / 1.5295–1.5938 | 166.033 |
| replay | mjwarp | 16 | 1 | passed | 4.0329 / 4.0266–4.0432 | 3.967 |
| replay | mjwarp | 32 | 1 | passed | 4.2905 / 4.2656–4.2920 | 7.458 |
| replay | mjwarp | 2048 | 1 | failed | — | — |
| replay | mujoco | 1024 | 32 | passed | 5.8830 / 5.7947–5.9306 | 174.061 |
| replay | mujoco | 64 | 32 | passed | 0.3749 / 0.3719–0.3776 | 170.703 |
| replay | mjwarp | 64 | 1 | passed | 4.5911 / 4.5786–4.5920 | 13.940 |
| replay | mujoco | 128 | 32 | passed | 0.7519 / 0.7451–0.7608 | 170.237 |
| replay | mujoco | 32 | 32 | passed | 0.2139 / 0.2116–0.2193 | 149.582 |
| replay | mjwarp | 1 | 1 | passed | 3.2123 / 3.1947–3.2226 | 0.311 |
| replay | mjwarp | 512 | 1 | failed | — | — |
| replay | mujoco | 512 | 32 | passed | 2.9624 / 2.9277–2.9831 | 172.834 |

Replay: episode reset and precomputed task controls, with full-state observations recorded every physics step. Model setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. CPU writes float64 trajectories to RAM; GPU writes float32 trajectories to VRAM. Successful tasks/s is validated episode replay throughput, not policy inference, learning speed or rendering performance.

The CPU baseline uses the configured serial or persistent worker pool. The full sweep defaults to one CPU pool case per batch; select --cpu-baseline both to add serial cases.

## Native MuJoCo CPU vs MuJoCo Warp replay

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.055× | 0.055× |
| 16 | 16 / 16 | 0.049× | 0.050× |
| 32 | 32 / 32 | 0.050× | 0.053× |
| 64 | 32 / 32 | 0.082× | 0.087× |
| 128 | 32 / 32 | 0.153× | 0.162× |
| 256 | 32 / 32 | 0.289× | 0.302× |
| 1024 | 32 / 32 | 0.902× | 0.882× |

No GPU simulation advantage was measured at the tested batch sizes.

No GPU advantage including output transfer and validation was measured at the tested batch sizes.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
