# Task-specific migration benchmark

Community measurements on the recorded hardware; not an official product benchmark.

Run status: **incomplete_or_failed**. Robot: **rebot**.

| Scope | Backend | Worlds | CPU threads | Status | Median / min–max (s) | Successful tasks/s |
|---|---|---:|---:|---|---|---:|
| replay | mjwarp | 256 | 1 | failed | — | — |
| replay | mjwarp | 128 | 1 | failed | — | — |
| replay | mujoco | 1 | 1 | passed | 1.0518 / 1.0493–1.1835 | 0.951 |
| replay | mujoco | 2048 | 32 | passed | 79.1299 / 79.0288–79.3944 | 25.881 |
| replay | mujoco | 16 | 16 | passed | 1.3047 / 1.2880–1.3540 | 12.263 |
| replay | mjwarp | 1024 | 1 | failed | — | — |
| replay | mujoco | 256 | 32 | passed | 10.3190 / 10.2685–10.3521 | 24.809 |
| replay | mjwarp | 16 | 1 | passed | 21.0475 / 20.9841–21.1850 | 0.760 |
| replay | mjwarp | 32 | 1 | passed | 22.5130 / 22.4365–22.5692 | 1.421 |
| replay | mjwarp | 2048 | 1 | failed | — | — |
| replay | mujoco | 1024 | 32 | passed | 39.8046 / 39.6907–39.8496 | 25.726 |
| replay | mujoco | 64 | 32 | passed | 2.5379 / 2.5297–2.5570 | 25.217 |
| replay | mjwarp | 64 | 1 | passed | 23.7271 / 23.6611–23.7700 | 2.697 |
| replay | mujoco | 128 | 32 | passed | 5.0683 / 5.0103–5.0887 | 25.255 |
| replay | mujoco | 32 | 32 | passed | 1.2857 / 1.2786–1.3118 | 24.889 |
| replay | mjwarp | 1 | 1 | passed | 18.6417 / 18.5038–18.7723 | 0.054 |
| replay | mjwarp | 512 | 1 | passed | 30.1455 / 30.0920–30.1740 | 16.984 |
| replay | mujoco | 512 | 32 | passed | 20.2056 / 20.1075–20.3418 | 25.339 |

Two-cube box task: each episode runs all 40 seconds (2000 control frames, 40000 physics steps). Episode reset, precomputed controls, a backend forward pass and actual solved-contact observations at 50 Hz are inside the simulation timer. Both backends record the same 61 float32 fields. Native MuJoCo uses persistent independent worker processes; GPU uses batched CUDA graphs. Every cube in every world must pass grasp, lift, carry, detached whole-cube containment and settling gates. Setup, control generation, warm-up, final output transfer and acceptance checks are outside the simulation timer and reported separately. Successful tasks/s measures checked episode throughput, not policy inference, training or rendering.

The full sweep defaults to one CPU pool case per batch; select --cpu-baseline both to add serial cases.

## Native MuJoCo CPU vs MuJoCo Warp replay

| Worlds | CPU threads (simulation / checked results) | GPU simulation speedup | GPU checked-results speedup |
|---:|---:|---:|---:|
| 1 | 1 / 1 | 0.056× | 0.061× |
| 16 | 16 / 16 | 0.062× | 0.066× |
| 32 | 32 / 32 | 0.057× | 0.063× |
| 64 | 32 / 32 | 0.107× | 0.115× |
| 512 | 32 / 32 | 0.670× | 0.686× |

No GPU simulation advantage was measured at the tested batch sizes.

No GPU advantage including output transfer and validation was measured at the tested batch sizes.

Hardware, concurrent load, exact dependency versions, source/model/control hashes, every repetition, separate costs and capacity/convergence diagnostics are in `results.json` and the individual case JSON files. Missing or failed cases are never replaced with estimates. Validate the same workload on a developer-accessible GPU before generalizing workstation results.
