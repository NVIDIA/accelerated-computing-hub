## When is migration worth it?

We compared Newton SolverMuJoCo on CPU and CUDA on the same task with SO-101 and reBot: pick up both cubes, place them in the receiving box, and withdraw the gripper. Each world completes 40 simulated seconds. We tested nine batch sizes, from one to 2048 independent worlds.

The table shows median simulation seconds across five repetitions after one full warm-up, measured on shared AMD Ryzen Threadripper PRO 9985WX 64-Cores and one NVIDIA RTX PRO 6000 Blackwell Workstation Edition. The CPU uses up to 32 persistent workers, capped by world count; the GPU uses one device. Reset, commands, integration and grasp-verification observations are included. Inverse kinematics is precomputed before timing and replayed identically on both backends.

| Worlds | SO-101 CPU (s) | SO-101 GPU (s) | reBot CPU (s) | reBot GPU (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.4848 | 14.7928 | 6.3294 | 15.5382 |
| 16 | 7.2303 | 16.7782 | 7.4193 | 17.1097 |
| 32 | 7.6543 | 17.2842 | 7.4869 | 17.6607 |
| 64 | 15.2200 | 17.8851 | 15.1986 | 18.6563 |
| 128 | 30.4011 | 18.7815 | 29.9407 | 19.3498 |
| 256 | 60.7351 | 19.9407 | 59.7683 | 20.6777 |
| 512 | 122.6874 | 21.6250 | 119.8159 | 22.3135 |
| 1024 | 242.4549 | 23.8462 | 238.9082 | Excluded |
| 2048 | 482.4144 | 28.3054 | 475.6768 | 30.0213 |

For SO-101, the first tested batch faster on GPU was 128 worlds; the CPU took 17.04 times as long at 2048 worlds. For reBot, the first tested batch faster on GPU was 128 worlds; the CPU took 15.84 times as long at 2048 worlds. These observations apply to the tested sizes, rather than universal crossover thresholds.

Including host output collection and task validation: For SO-101, the first tested batch faster on GPU was 128 worlds; the CPU took 14.00 times as long at 2048 worlds. For reBot, the first tested batch faster on GPU was 128 worlds; the CPU took 13.19 times as long at 2048 worlds. These medians sum all three costs within each repetition. Setup, compilation, warm-up and validator teardown remain separate.

Excluded configurations: reBot GPU at 1024 worlds. 35 of 36 configurations passed every required episode. Every accepted world must physically grasp, lift, carry, release and settle both cubes inside the box. One failed episode excludes the entire configuration and its timing from comparisons; exact physical and numerical checks are in the guide.

Use the [benchmark notebook](https://github.com/johnnynunez/blogs/blob/feature/blog3-validated-gpu-box/Article_3/part3/05_Notebook_Migration_Benchmark.ipynb) to reproduce the task. The [companion benchmark guide](https://github.com/johnnynunez/blogs/blob/feature/blog3-validated-gpu-box/Article_3/BENCHMARK.md) explains the measurement method; [full results](https://github.com/johnnynunez/blogs/blob/feature/blog3-validated-gpu-box/Article_3/benchmark-results/2026-10-09-validated-box-gpu0/README.md) retain ratios, ranges, failures and hardware configuration. These community measurements are not official product benchmarks or evidence of lower-end GPU, Colab or Brev performance.

The companion also includes an [ALOHA pot-and-lid replay](https://github.com/johnnynunez/blogs/blob/feature/blog3-validated-gpu-box/Article_3/reference-benchmark/README.md) comparing native MuJoCo on CPU with MuJoCo Warp on GPU. It is a separate workload and does not measure Newton's API or this box task.
