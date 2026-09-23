# NVIDIA Warp Tutorial

This tutorial contains notebooks for learning [NVIDIA Warp](https://github.com/NVIDIA/warp), an open-source Python framework for writing high-performance simulation and graphics code on CPUs and NVIDIA GPUs.

These notebooks can be run on [NVIDIA Brev](https://brev.nvidia.com) or [Google Colab](https://colab.research.google.com).

- [Docker Images](https://github.com/NVIDIA/accelerated-computing-hub/pkgs/container/warp-tutorial) and [Docker Compose files](./brev/docker-compose.yml) for creating Brev Launchables or running locally.

Brev Launchables of this tutorial should use:
- L40S, L4, or T4 instances.
- Crusoe or any other provider with Flexible Ports.

## Notebooks

| Notebook | Description | Colab |
| --- | --- | --- |
| [01. Introduction to NVIDIA Warp](notebooks/01__intro_to_warp.ipynb) | Core Warp programming model, arrays, kernels, automatic differentiation, and a galaxy simulation capstone. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/01__intro_to_warp.ipynb) |
| [02. Ising Model](notebooks/02__ising_model.ipynb) | A 2D Ising model simulation built with Warp kernels and GPU arrays. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/02__ising_model.ipynb) |
| [03. Navier-Stokes Solver](notebooks/03__navier_stokes_solver.ipynb) | Builds a 2D Navier-Stokes solver using Warp kernels, tiled FFTs, and CUDA graph capture. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/03__navier_stokes_solver.ipynb) |
| [04. Differentiable Navier-Stokes Solver](notebooks/04__differentiable_navier_stokes_solver.ipynb) | Uses Warp automatic differentiation to optimize Navier-Stokes simulation inputs. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/04__differentiable_navier_stokes_solver.ipynb) |

### MuJoCo and MuJoCo Warp robotics lessons

The [SO-101 simulation companion](notebooks/mujoco/README.md) develops a CPU
pick-and-place task, migrates its physics to MuJoCo Warp, and measures batched
GPU stepping. It includes exercises, completed solutions, a pinned `uv`
environment, and task validation.

| Notebook | Description | Colab |
| --- | --- | --- |
| [05. MuJoCo Fundamentals](notebooks/mujoco/part1/05__mujoco_fundamentals.ipynb) | Load the SO-101 model, inspect state, and control a CPU simulation. | [Open in Colab](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/mujoco/part1/05__mujoco_fundamentals.ipynb) |
| [06. Pick and Place](notebooks/mujoco/part1/06__pick_and_place.ipynb) | Complete the physics loop and validate a two-cube stacking task. | [Open in Colab](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/mujoco/part1/06__pick_and_place.ipynb) |
| [07. MuJoCo Warp](notebooks/mujoco/part2/07__mujoco_warp.ipynb) | Seed GPU state, validate one world, capture CUDA work, and benchmark a batch. | [Open in Colab](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/mujoco/part2/07__mujoco_warp.ipynb) |

## Layout

Notebook-specific images are stored under `notebooks/images/` in subdirectories named for each notebook. This keeps assets organized as more Warp notebooks are added.

## Additional Resources

- [NVIDIA Warp documentation](https://nvidia.github.io/warp/)
- [NVIDIA Warp GitHub repository](https://github.com/NVIDIA/warp)
- [Warp example gallery](https://github.com/NVIDIA/warp?tab=readme-ov-file#running-examples)
