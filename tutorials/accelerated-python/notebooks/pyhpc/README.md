# PyHPC Course

This Accelerated Python course tours the high-performance Python landscape: the NumPy and CuPy array model, distributed computing with mpi4py, alternative programming models and Python/C++ interoperability, and authoring your own CUDA kernels. Along the way it solves the same 1D Shallow Water Equations bump pulse end to end with JAX, PyOMP, nanobind, CppJIT, and mpi4py, each measured against a NumPy baseline, and profiles real kernels with NVIDIA's developer tools.

- [Notebooks](.) containing lessons and exercises, intended for self-paced or instructor-led learning, which can be run on [NVIDIA Brev](https://brev.nvidia.com), locally with Docker, or on CSCS Daint. Self-contained lessons that show a Colab badge can also run on [Google Colab](https://colab.research.google.com); the interoperability and SWE lessons require the shared tutorial image and source files.
- [Syllabi](../syllabi) that select a subset of the Accelerated Python notebooks for a particular learning objective.
- [Docker Compose file](../../brev/docker-compose.yml) for creating a Brev Launchable or running locally.

Brev Launchables of this tutorial should use:
- L40S, L4, or T4 instances. The mpi4py notebook runs multiple local CPU ranks.
- A recent NVIDIA driver. The image ships a CUDA 13.2.1 toolkit, so the host driver must support CUDA 13.
- Crusoe or any other provider with Flexible Ports.

## Syllabi

- [PyHPC - CuPy, Kernels, MPI, JAX, OMP, Interop - 2 Days](../syllabi/pyhpc__cupy_kernels_mpi_jax_omp_interop__2_days.ipynb)

## Notebooks

Each exercise notebook that has a paired solution carries `# TODO:` cells with `...` placeholders; the solution fills them in. The intro/reference notebook (08) is complete as written and has no separate solution.

### Fundamentals

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 00 | NumPy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/00__numpy.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/solutions/00__numpy__SOLUTION.ipynb) |
| 01 | CuPy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/01__cupy.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/solutions/01__cupy__SOLUTION.ipynb) |
| 02 | Power Iteration - CuPy - Memory Spaces | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/02__power_iteration__cupy__memory_spaces.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/solutions/02__power_iteration__cupy__memory_spaces__SOLUTION.ipynb) |

### Kernels

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 03 | Power Iteration - CuPy - Asynchrony | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/03__power_iteration__cupy__asynchrony.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/solutions/03__power_iteration__cupy__asynchrony__SOLUTION.ipynb) |
| 04 | Copy - Kernel Authoring | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/04__copy__kernel_authoring.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/solutions/04__copy__kernel_authoring__SOLUTION.ipynb) |
| 05 | Book Histogram - Kernel Authoring | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/05__book_histogram__kernel_authoring.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/solutions/05__book_histogram__kernel_authoring__SOLUTION.ipynb) |

### Distributed

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 06 | mpi4py | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/06__mpi4py.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/pyhpc/solutions/06__mpi4py__SOLUTION.ipynb) |

### Python/C++ interoperability

A standalone comparison of several ways to call C and C++ from Python (ctypes, cffi, nanobind, and CppJIT), benchmarked on the different kernels that expose various C++ features.

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 07 | C++ Interop: ctypes, cffi, nanobind, and CppJIT compared | [Open](./07__cpp_interop.ipynb) | [Solution](./solutions/07__cpp_interop__SOLUTION.ipynb) |

### Programming models and interoperability

Notebooks 08-14 are the Shallow Water Equations "ladder": an intro plus NumPy baseline, five solvers that each re-implement the same timestep with a different tool, and a synthesis that reads the measured timings and compares them. The synthesis notebook collects the per-tool rows from `timings.json` (written by notebooks 08 to 13) and closes with a matched-precision float64 comparison across the memory hierarchy.

| # | Notebook | Link | Solution |
|---|----------|------|----------|
| 08 | SWE - Intro | [Open](./08__swe__intro.ipynb) |  |
| 09 | SWE - JAX | [Open](./09__swe__jax.ipynb) | [Solution](./solutions/09__swe__jax__SOLUTION.ipynb) |
| 10 | SWE - PyOMP | [Open](./10__swe__pyomp.ipynb) | [Solution](./solutions/10__swe__pyomp__SOLUTION.ipynb) |
| 11 | SWE - nanobind | [Open](./11__swe__nanobind.ipynb) | [Solution](./solutions/11__swe__nanobind__SOLUTION.ipynb) |
| 12 | SWE - CppJIT - CUB | [Open](./12__swe__cppjit__cub.ipynb) | [Solution](./solutions/12__swe__cppjit__cub__SOLUTION.ipynb) |
| 13 | SWE - mpi4py | [Open](./13__swe__mpi4py.ipynb) | [Solution](./solutions/13__swe__mpi4py__SOLUTION.ipynb) |
| 14 | SWE - Synthesis | [Open](./14__swe__synthesis.ipynb) | [Solution](./solutions/14__swe__synthesis__SOLUTION.ipynb) |

## The Shallow Water Equations problem

A 1D shallow-water bump pulse: a small mound of water at rest splits into two outgoing wave packets. Two conserved fields (`h`, `hu`) advance under a forward-Euler step. This PDE is small enough to read in full while exhibiting nonlinearity, and a fixed number of steps from the initial condition gives a result we can compare across tools. The full specification is in [`08__swe__intro.ipynb`](./08__swe__intro.ipynb).

## Running

On Brev, deploy the Accelerated Python [Docker Compose file](../../brev/docker-compose.yml) as a Launchable and open the JupyterLab port.

On CSCS Alps/Daint, follow the [CSCS launch guide](../../../../docs/cscs.md) to run JupyterLab and both Nsight Streamers through Slurm Container Engine.

Locally, with an NVIDIA GPU and a CUDA-13-capable driver:

```bash
docker compose -f tutorials/accelerated-python/brev/docker-compose.yml up
```

Then open JupyterLab on port 8888. The notebooks are self-contained and can be run in any order, with one exception: the Shallow Water Equations ladder writes `timings.json` as you run notebooks 08 to 13, so run those before the synthesis notebook (14).

The image runs every notebook with the same system Python packages. The
`Python 3 (PyHPC)` and Nsight profiler kernels differ only in their kernel-local
configuration: `IPYTHONDIR=/opt/pyhpc-ipython` loads the course's startup files,
`MPI4PY_MPIABI=mpich` selects the MPICH module from the multi-ABI mpi4py wheel,
and `PATH=/opt/pyhpc-mpi/bin:${PATH}` selects the matching launchers. The default
Python environment continues to select OpenMPI, so existing Accelerated Python
notebooks keep their expected MPI runtime.

## CppJIT toolchain

Notebook 12 uses CppJIT to automatically bind our Python runtime with CUDA C++, using the clang-repl C++ interpreter and [CppInterOp](https://github.com/compiler-research/CppInterOp). [CppJIT](https://github.com/compiler-research/CppJIT) is the successor to the [cppyy](https://cppyy.readthedocs.io/) automatic bindings tool. Although an alpha release is available on PyPI, this tutorial source-builds the patched ISC 2026 branch it requires, so the notebook currently runs only in the tutorial image.
