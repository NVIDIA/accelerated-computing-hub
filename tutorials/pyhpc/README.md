# PyHPC Tutorial

This tutorial tours the high-performance Python landscape: the NumPy and CuPy array model, distributed computing with mpi4py, alternative programming models and Python/C++ interoperability, and authoring your own CUDA kernels. Along the way it solves the same 1D Shallow Water Equations bump pulse end to end with JAX, PyOMP, nanobind, and CppJIT, each measured against a NumPy baseline, and profiles real kernels with NVIDIA's developer tools.

- [Notebooks](./notebooks) containing lessons and exercises, intended for self-paced or instructor-led learning, which can be run on [NVIDIA Brev](https://brev.nvidia.com), locally with Docker, or on [Google Colab](https://colab.research.google.com).
- [Syllabi](./notebooks/syllabi) that select a subset of the notebooks for a particular learning objective.
- [Docker Compose file](./brev/docker-compose.yml) for creating a Brev Launchable or running locally.

Brev Launchables of this tutorial should use:
- L40S, L4, or T4 instances. The mpi4py notebook runs multiple local CPU ranks.
- A recent NVIDIA driver. The image ships a CUDA 13.1 toolkit, so the host driver must support CUDA 13.
- Crusoe or any other provider with Flexible Ports.

## Syllabi

- [PyHPC - CuPy, mpi4py, JAX, CppInterOp, & CUDA Kernels - 8 Hours](./notebooks/syllabi/pyhpc__cupy_mpi4py_jax_cppinterop_kernels__8_hours.ipynb)

## Notebooks

Each exercise notebook that has a paired solution carries `# TODO:` cells with `...` placeholders; the solution fills them in. The intro/reference notebook (04), the synthesis notebook (09), and the mpi4py walkthrough (03) are complete as written and have no separate solution.

### Fundamentals

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 00 | NumPy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/00__numpy.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/00__numpy__SOLUTION.ipynb) |
| 01 | CuPy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/01__cupy.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/01__cupy__SOLUTION.ipynb) |
| 02 | Power Iteration - CuPy - Memory Spaces | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/02__power_iteration__cupy__memory_spaces.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/02__power_iteration__cupy__memory_spaces__SOLUTION.ipynb) |

### Distributed

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 03 | mpi4py | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/03__mpi4py.ipynb) |  |

### Programming models and interoperability

Notebooks 04-09 are the Shallow Water Equations "ladder": an intro plus NumPy baseline, four solvers that each re-implement the same timestep with a different tool, and a synthesis that reads the measured timings and compares them. The synthesis notebook collects the per-tool rows from `timings.json` (written by notebooks 04 to 08) and closes with a matched-precision GPU comparison of the JAX and CppJIT device paths.

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 04 | SWE - Intro | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/04__swe__intro.ipynb) |  |
| 05 | SWE - JAX | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/05__swe__jax.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/05__swe__jax__SOLUTION.ipynb) |
| 06 | SWE - PyOMP | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/06__swe__pyomp.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/06__swe__pyomp__SOLUTION.ipynb) |
| 07 | SWE - nanobind | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/07__swe__nanobind.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/07__swe__nanobind__SOLUTION.ipynb) |
| 08 | SWE - CppJIT - Thrust | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/08__swe__cppjit__thrust.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/08__swe__cppjit__thrust__SOLUTION.ipynb) |
| 09 | SWE - Synthesis | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/09__swe__synthesis.ipynb) |  |

### Kernels

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 10 | Power Iteration - CuPy - Asynchrony | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/10__power_iteration__cupy__asynchrony.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/10__power_iteration__cupy__asynchrony__SOLUTION.ipynb) |
| 11 | Copy - Kernel Authoring | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/11__copy__kernel_authoring.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/11__copy__kernel_authoring__SOLUTION.ipynb) |
| 12 | Book Histogram - Kernel Authoring | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/12__book_histogram__kernel_authoring.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/12__book_histogram__kernel_authoring__SOLUTION.ipynb) |

### Python/C++ interoperability

A standalone comparison of several ways to call C and C++ from Python (ctypes, cffi, nanobind, and CppJIT), benchmarked on the different kernels that expose various C++ features.

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 13 | C++ Interop: ctypes, cffi, nanobind, and CppJIT compared | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/13__cpp_interop.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/13__cpp_interop__SOLUTION.ipynb) |

## The Shallow Water Equations problem

A 1D shallow-water bump pulse: a small mound of water at rest splits into two outgoing wave packets. Two conserved fields (`h`, `hu`) advance under a forward-Euler step. This PDE is small enough to read in full while exhibiting nonlinearity, and a fixed number of steps from the initial condition gives a result we can compare across tools. The full specification is in [`04__swe__intro.ipynb`](./notebooks/04__swe__intro.ipynb).

## Running

On Brev, deploy the [Docker Compose file](./brev/docker-compose.yml) as a Launchable and open the JupyterLab port.

Locally, with an NVIDIA GPU and a CUDA-13-capable driver:

```bash
docker compose -f tutorials/pyhpc/brev/docker-compose.yml up
```

Then open JupyterLab on port 8888. The notebooks are self-contained and can be run in any order, with one exception: the Shallow Water Equations ladder writes `timings.json` as you run notebooks 04 to 08, so run those before the synthesis notebook (09).

## CppJIT toolchain

Notebook 08 uses CppJIT to automatically bind our Python runtime with CUDA C++, using the clang-repl C++ interpreter and [CppInterOp](https://github.com/compiler-research/CppInterOp). This is currently source built in the [Docker image](./brev/dockerfile) and not a standard `pip install`, so this will only run in the tutorial image. [CppJIT](https://github.com/compiler-research/CppJIT) is the successor to the [cppyy](https://cppyy.readthedocs.io/) automatic bindings tool, and no official CppJIT release is published on PyPI yet (beta release planned for end of summer 2026).
