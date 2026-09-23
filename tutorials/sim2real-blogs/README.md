# Sim2Real Blogs

Companion tutorials for the simulation-to-real robotics blog series. Each article
has its own lessons, exercises, and completed solutions.

## Articles

| Article | Lessons |
| --- | --- |
| 2. How to Use NVIDIA Warp and MJWarp to Accelerate Robotics Simulation and Learning Workflows | [SO-101 simulation: from MuJoCo to MuJoCo Warp](notebooks/mujoco/README.md) |

The current lessons cover simulation and physics throughput; they do not deploy
a policy to a physical robot.

## Notebooks

| Notebook | Description | Colab |
| --- | --- | --- |
| [01. MuJoCo Fundamentals](notebooks/mujoco/part1/01__mujoco_fundamentals.ipynb) | Load the SO-101 model, inspect state, and control a CPU simulation. | [Open in Colab](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/sim2real-blogs/notebooks/mujoco/part1/01__mujoco_fundamentals.ipynb) |
| [02. Pick and Place](notebooks/mujoco/part1/02__pick_and_place.ipynb) | Complete the physics loop and validate a two-cube stacking task. | [Open in Colab](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/sim2real-blogs/notebooks/mujoco/part1/02__pick_and_place.ipynb) |
| [03. MuJoCo Warp](notebooks/mujoco/part2/03__mujoco_warp.ipynb) | Seed GPU state, validate one world, capture CUDA work, and benchmark a batch. | [Open in Colab](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/sim2real-blogs/notebooks/mujoco/part2/03__mujoco_warp.ipynb) |

Follow the [local Python 3.12 setup and validation instructions](notebooks/mujoco/README.md)
to run the lessons. Colab links target upstream `main` and become available after merge.

## Brev and Docker

This tutorial has its own [Dockerfile](brev/dockerfile),
[dependencies](brev/requirements.txt), [Docker Compose configuration](brev/docker-compose.yml),
and [test entrypoint](brev/test.bash). Use an NVIDIA CUDA GPU such as an L40S, L4,
or T4 and a provider with Flexible Ports for Brev.

From the repository root, build and run the dedicated environment:

```bash
./brev/dev-build.bash sim2real-blogs
./brev/dev-start.bash sim2real-blogs
./brev/dev-test.bash sim2real-blogs
```

The Compose configuration uses `ghcr.io/nvidia/sim2real-blogs-tutorial:latest`;
build locally until that image is published by the repository's CI.

## Attribution

The imported lessons retain their [MIT license](notebooks/mujoco/LICENSE).
See the [third-party notices](notebooks/mujoco/THIRD_PARTY_NOTICES.md) for
source attribution and robot asset licenses.
