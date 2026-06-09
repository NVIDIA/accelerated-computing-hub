# NVIDIA Warp Tutorial

This tutorial contains notebooks for learning [NVIDIA Warp](https://github.com/NVIDIA/warp), an open-source Python framework for writing high-performance simulation and graphics code on CPUs and NVIDIA GPUs.

## Notebooks

| Notebook | Description | Colab |
| --- | --- | --- |
| [01. Introduction to NVIDIA Warp](notebooks/01_intro_to_warp.ipynb) | Core Warp programming model, arrays, kernels, automatic differentiation, and a galaxy simulation capstone. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/01_intro_to_warp.ipynb) |
| [02. Ising Model](notebooks/02_ising_model.ipynb) | A 2D Ising model simulation built with Warp kernels and GPU arrays. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/02_ising_model.ipynb) |
| [03. Navier-Stokes Solver](notebooks/03_navier_stokes_solver.ipynb) | Builds a 2D Navier-Stokes solver using Warp kernels, tiled FFTs, and CUDA graph capture. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/03_navier_stokes_solver.ipynb) |
| [04. Differentiable Navier-Stokes Solver](notebooks/04_differentiable_navier_stokes_solver.ipynb) | Uses Warp automatic differentiation to optimize Navier-Stokes simulation inputs. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/warp/notebooks/04_differentiable_navier_stokes_solver.ipynb) |

## Layout

Notebook-specific images are stored under `notebooks/images/` in subdirectories named for each notebook. This keeps assets organized as more Warp notebooks are added.

## Additional Resources

- [NVIDIA Warp documentation](https://nvidia.github.io/warp/)
- [NVIDIA Warp GitHub repository](https://github.com/NVIDIA/warp)
- [Warp example gallery](https://github.com/NVIDIA/warp?tab=readme-ov-file#running-examples)
