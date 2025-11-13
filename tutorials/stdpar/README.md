# Portable GPU Acceleration with Standard Parallelism

This tutorial teaches you how to accelerate portable HPC applications with CPUs and GPUs using the parallelism and concurrency features of modern Standard C++ and Fortran standards. You'll find the following content:

- [Notebooks](./notebooks) containing lessons and exercises, intended for self-paced or instructor-led learning, which can be run on [NVIDIA Brev](https://brev.nvidia.com) or [Google Colab](https://colab.research.google.com).
- [Slides](./slides) containing the lecture content for the lessons.
- [Docker Images](https://github.com/NVIDIA/accelerated-computing-hub/pkgs/container/stdpar-tutorial) and [Docker Compose files](./brev/docker-compose.yml) for creating Brev Launchables or running locally.

Brev Launchables of this tutorial should use:
- 4xL4, 2xL4, 2xL40S, or 1x L40S instances.
- GCP, AWS, or any other with Flexible Ports and Linux 6.1.24+, 6.2.11+, or 6.3+ (for HMM).

## Notebooks

### Portable GPU Acceleration of HPC Applications with ISO C++

| Notebook                                | Link                                                                                                                                                                                       |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/cpp/start.ipynb) |
| Lab 1: DAXPY | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/cpp/lab1_daxpy/daxpy.ipynb) |
| Lab 1: Select (optional) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/cpp/lab1_select/select.ipynb) |
| Lab 2: 2D Heat Equation | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/cpp/lab2_heat/heat.ipynb) |
| Lab 3: Parallel Tree Construction | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/cpp/lab3_tree/tree.ipynb) |

### Accelerating Portable HPC Applications with ISO Fortran

| Notebook                                | Link                                                                                                                                                                                       |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/fortran/start.ipynb) |
| Lab 1: MATMUL | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/fortran/lab1_matmul/matmul.ipynb) |
| Lab 2: DAXPY | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/fortran/lab2_daxpy/daxpy.ipynb) |
| Lab 3: Heat Equation | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/stdpar/notebooks/fortran/lab3_heat/heat.ipynb) |
