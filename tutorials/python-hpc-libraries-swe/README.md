# Python HPC Libraries (SWE) Tutorial

This tutorial tours the part of the HPC Python landscape beyond array libraries: reaching for a different programming model, or dropping into a C/C++ kernel to integrate with code you already maintain. The same 1D Shallow Water Equations bump pulse is solved end to end with JAX, PyOMP, nanobind, and CppJIT, each measured against a NumPy baseline on the same problem.

- [Notebooks](./notebooks) containing lessons and exercises, intended for self-paced or instructor-led learning, which can be run on [NVIDIA Brev](https://brev.nvidia.com) or locally with Docker.
- [Docker Compose file](./brev/docker-compose.yml) for creating a Brev Launchable or running locally.

Brev Launchables of this tutorial should use:
- A single L40S, L4, or RTX GPU. The problem runs on one device.
- A recent NVIDIA driver. The image ships a CUDA 13.1 toolkit, so the host driver must support CUDA 13.
- Crusoe or any other provider with Flexible Ports.

## The problem

A 1D shallow-water bump pulse: a small mound of water at rest splits into two outgoing wave packets. Two conserved fields (`h`, `hu`) advance under a forward-Euler step. This PDE is small enough to read in full while exhibiting nonlinearity, and a fixed number of steps from the initial condition gives a result we can compare across tools. The full specification is in `00__intro.ipynb`.

## Notebooks

The notebooks build in order: a NumPy baseline, four solvers that each re-implement the same timestep with a different tool, and a synthesis that reads the measured timings and compares them. Each exercise notebook from 02 onward has a paired solution. The exercise version carries `# TODO:` cells with `...` placeholders, and the solution fills them in.

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 00 | Intro: the problem and the tool ladder | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/00__intro.ipynb) | |
| 01 | Reference solver: baseline in NumPy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/01__swe_core__reference_solver.ipynb) | |
| 02 | JAX: `jit` + `lax.scan` | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/02__jax__.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/solutions/02__jax__SOLUTION.ipynb) |
| 03 | PyOMP: `#pragma omp parallel for` from Python | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/03__pyomp__.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/solutions/03__pyomp__SOLUTION.ipynb) |
| 04 | nanobind: bind a hand-written C++ kernel | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/04__nanobind__.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/solutions/04__nanobind__SOLUTION.ipynb) |
| 05 | CppJIT: live CUDA in a notebook | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/05__cppjit__.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/solutions/05__cppjit__SOLUTION.ipynb) |
| 06 | Synthesis: runtime, throughput, and footprint | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/python-hpc-libraries-swe/notebooks/06__synthesis.ipynb) | |

The synthesis notebook collects the per-tool rows from `timings.json` (written by notebooks 01 to 05) and closes with a matched-precision GPU comparison of the JAX and CppJIT device paths.

## Running

On Brev, deploy the [Docker Compose file](./brev/docker-compose.yml) as a Launchable and open the JupyterLab port.

Locally, with an NVIDIA GPU and a CUDA-13-capable driver:

```bash
docker compose -f tutorials/python-hpc-libraries-swe/brev/docker-compose.yml up
```

Then open JupyterLab on port 8888 and run the notebooks in order. `timings.json` is written as you run notebooks 01 to 05, so run those before the synthesis notebook.

## CppJIT toolchain

Notebook 05 uses CppJIT to automatically bind our Python runtime with CUDA C++, using the clang-repl C++ interpreter and [CppInterOp](https://github.com/compiler-research/CppInterOp). This is currently source built in the [Docker image](./brev/dockerfile) and not a standard `pip install`, so this will only run in the tutorial image. [CppJIT](https://github.com/compiler-research/CppJIT) is the successor to the [cppyy](https://cppyy.readthedocs.io/) automatic bindings tool, and no official CppJIT release is published on PyPI yet.
