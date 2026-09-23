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

## Layout

Notebook-specific images are stored under `notebooks/images/` in subdirectories named for each notebook. This keeps assets organized as more Warp notebooks are added.

## Additional Resources

- [NVIDIA Warp documentation](https://nvidia.github.io/warp/)
- [NVIDIA Warp GitHub repository](https://github.com/NVIDIA/warp)
- [Warp example gallery](https://github.com/NVIDIA/warp?tab=readme-ov-file#running-examples)
