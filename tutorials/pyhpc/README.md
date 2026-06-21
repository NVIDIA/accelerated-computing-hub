# PyHPC Tutorial

This tutorial tours the high-performance Python landscape: the NumPy and CuPy array model, distributed computing with mpi4py, alternative programming models and Python/C++ interoperability, and authoring your own CUDA kernels. Along the way it solves the same 1D Shallow Water Equations bump pulse end to end with JAX, PyOMP, nanobind, and CppJIT, each measured against a NumPy baseline, and profiles real kernels with NVIDIA's developer tools.

- [Notebooks](./notebooks) containing lessons and exercises, intended for self-paced or instructor-led learning, which can be run on [NVIDIA Brev](https://brev.nvidia.com), locally with Docker, or on [Google Colab](https://colab.research.google.com).
- [Syllabi](./notebooks/syllabi) that select a subset of the notebooks for a particular learning objective.
- [Docker Compose file](./brev/docker-compose.yml) for creating a Brev Launchable or running locally.

Brev Launchables of this tutorial should use:
- L40S, L4, or T4 instances for the single-GPU notebooks.
- 4xL4, 2xL4, or 2xL40S instances to give the distributed (mpi4py) notebook multiple ranks.
- A recent NVIDIA driver. The image ships a CUDA 13.1 toolkit, so the host driver must support CUDA 13.
- Crusoe or any other provider with Flexible Ports.

## Syllabi

- [PyHPC - NumPy, CuPy, JAX, mpi4py, & CUDA Kernels - 8 Hours](./notebooks/syllabi/pyhpc__numpy_cupy_jax_mpi4py_kernels__8_hours.ipynb)

## Notebooks

Each exercise notebook that has a paired solution carries `# TODO:` cells with `...` placeholders; the solution fills them in. The framing notebooks (04, 10), the NumPy reference (05), and the mpi4py walkthrough (03) are complete as written and have no separate solution.

### Fundamentals

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 00 | NumPy Intro: `ndarray` basics | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/00__numpy_intro__ndarray_basics.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/00__numpy_intro__ndarray_basics__SOLUTION.ipynb) |
| 01 | NumPy to CuPy: `ndarray` basics on the GPU | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/01__numpy_to_cupy__ndarray_basics.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/01__numpy_to_cupy__ndarray_basics__SOLUTION.ipynb) |
| 02 | Memory spaces: host vs. device (power iteration) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/02__memory_spaces__power_iteration.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/02__memory_spaces__power_iteration__SOLUTION.ipynb) |

### Distributed

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 03 | mpi4py: ranks, point-to-point, and collectives | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/03__mpi4py__collectives.ipynb) |  |

### Programming models and interoperability

Notebooks 04-10 are the Shallow Water Equations "ladder": a NumPy baseline, four solvers that each re-implement the same timestep with a different tool, and a synthesis that reads the measured timings and compares them. The synthesis notebook collects the per-tool rows from `timings.json` (written by notebooks 05 to 09) and closes with a matched-precision GPU comparison of the JAX and CppJIT device paths.

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 04 | Intro: the problem and the tool ladder | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/04__intro.ipynb) |  |
| 05 | Reference solver: baseline in NumPy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/05__swe_core__reference_solver.ipynb) |  |
| 06 | JAX: Python JIT compilation + whole-program optimization on device | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/06__jax__lax_scan.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/06__jax__lax_scan__SOLUTION.ipynb) |
| 07 | PyOMP: `#pragma omp parallel for` from Python | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/07__pyomp__parallel_for.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/07__pyomp__parallel_for__SOLUTION.ipynb) |
| 08 | nanobind: Expose C++ to Python with manual bindings | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/08__nanobind__cpp_kernel.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/08__nanobind__cpp_kernel__SOLUTION.ipynb) |
| 09 | CppJIT: Automatic Python/C++ Interoperability + GPU support | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/09__cppjit__gpu_thrust.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/09__cppjit__gpu_thrust__SOLUTION.ipynb) |
| 10 | Synthesis: runtime, throughput, and footprint | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/10__synthesis.ipynb) |  |

### Kernels

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 11 | Asynchrony: streams and overlap (power iteration) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/11__asynchrony__power_iteration.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/11__asynchrony__power_iteration__SOLUTION.ipynb) |
| 12 | Kernel authoring: copy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/12__kernel_authoring__copy.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/12__kernel_authoring__copy__SOLUTION.ipynb) |
| 13 | Kernel authoring: book histogram | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/13__kernel_authoring__book_histogram.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/pyhpc/notebooks/solutions/13__kernel_authoring__book_histogram__SOLUTION.ipynb) |

## The Shallow Water Equations problem

A 1D shallow-water bump pulse: a small mound of water at rest splits into two outgoing wave packets. Two conserved fields (`h`, `hu`) advance under a forward-Euler step. This PDE is small enough to read in full while exhibiting nonlinearity, and a fixed number of steps from the initial condition gives a result we can compare across tools. The full specification is in [`04__intro.ipynb`](./notebooks/04__intro.ipynb).

## Running

On Brev, deploy the [Docker Compose file](./brev/docker-compose.yml) as a Launchable and open the JupyterLab port.

Locally, with an NVIDIA GPU and a CUDA-13-capable driver:

```bash
docker compose -f tutorials/pyhpc/brev/docker-compose.yml up
```

Then open JupyterLab on port 8888. The notebooks are self-contained and can be run in any order, with one exception: the Shallow Water Equations ladder writes `timings.json` as you run notebooks 05 to 09, so run those before the synthesis notebook (10).

## CppJIT toolchain

Notebook 09 uses CppJIT to automatically bind our Python runtime with CUDA C++, using the clang-repl C++ interpreter and [CppInterOp](https://github.com/compiler-research/CppInterOp). This is currently source built in the [Docker image](./brev/dockerfile) and not a standard `pip install`, so this will only run in the tutorial image. [CppJIT](https://github.com/compiler-research/CppJIT) is the successor to the [cppyy](https://cppyy.readthedocs.io/) automatic bindings tool, and no official CppJIT release is published on PyPI yet (beta release planned for end of summer 2026)
