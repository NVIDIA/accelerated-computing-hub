# Accelerated Python Tutorial

This modular tutorial contains content on all things related to accelerated Python:

- [Notebooks](./notebooks) containing lessons and exercises organized by topic, including CUDA Tile and the PyHPC material on CuPy, CUDA kernels, MPI, JAX, PyOMP, and Python/C++ interoperability. They are intended for self-paced or instructor-led learning and can be run on [NVIDIA Brev](https://brev.nvidia.com) or [Google Colab](https://colab.research.google.com).
- [Slides](./slides) containing the lecture content for the lessons.
- [Syllabi](./notebooks/syllabi) that select a subset of the notebooks for a particular learning objective.
- [Docker Images](https://github.com/NVIDIA/accelerated-computing-hub/pkgs/container/accelerated-python-tutorial) and [Docker Compose files](./brev/docker-compose.yml) for creating Brev Launchables or running locally.

Brev Launchables of this tutorial should use:
- L40S, L4, or T4 instances (for non-distributed notebooks other than CUDA Tile).
- A10G or newer Ampere, Ada, or Blackwell instances for CUDA Tile notebooks.
- 4xL4 or 2xL4 instances (for distributed notebooks).
- A host driver with CUDA 13 support.
- Crusoe or any other provider with Flexible Ports.

## Syllabi

- [CUDA Python - CuPy, cuDF, CCCL, & Kernels - 8 Hours](./notebooks/syllabi/cuda_python__cupy_cudf_cccl_kernels__8_hours.ipynb).
- [CUDA Python - CuPy, Kernels, & cuTile - 8 Hours](./notebooks/syllabi/cuda_python__cupy_kernels_cutile__8_hours.ipynb).
- [CUDA Python - cuda.core & CCCL - 2 Hours](./notebooks/syllabi/cuda_python__cuda_core_cccl__2_hours.ipynb)
- [CUDA Tile - cuTile Python](./notebooks/syllabi/cutile.ipynb)
- [PyHPC - NumPy, CuPy, & mpi4py - 4 Hours](./notebooks/syllabi/pyhpc__numpy_cupy_mpi4py__4_hours.ipynb)
- [PyHPC - CuPy, Kernels, MPI, JAX, OMP, Interop - 2 Days](./notebooks/syllabi/pyhpc__cupy_kernels_mpi_jax_omp_interop__2_days.ipynb)

The two-day PyHPC syllabus and the eight-hour CUDA Python course select lessons
from the same topic directories as every other Accelerated Python course. The
eight-hour course contains only the CuPy and Numba CUDA kernel-authoring
foundations followed by CUDA Tile lessons 44 through 47. Self-contained lessons
with a Colab badge can run on Google Colab. C++ interoperability and the Shallow
Water Equations applications require the shared tutorial image and checked-in
source files.

Applications 81 through 87 form one ordered case study. They solve the same 1D
Shallow Water Equations problem with NumPy, JAX, PyOMP, nanobind, CppJIT/CUB,
and mpi4py. Notebooks 81 through 86 write measurements to `timings.json`; run
them before notebook 87, which compares the results. CppJIT is built from the
course's pinned ISC 2026 branch and is available in the tutorial image rather
than from the public alpha package.

The image has one system Python environment. The `Python 3 (PyHPC)` and Nsight
profiler kernels only select course-specific startup and MPI settings: MPICH is
used for local multi-rank exercises, while the default Python kernel continues
to use OpenMPI. For CSCS Alps/Daint deployment, follow the [CSCS launch
guide](../../docs/cscs.md).

## Upgrading an existing deployment

The merged course initializes a new Docker repository volume named
`accelerated-python_accelerated-computing-hub-v2`. Existing volumes named
`accelerated-python_accelerated-computing-hub` or
`pyhpc_accelerated-computing-hub` are deliberately left untouched. Before
removing either old volume, copy any edited notebooks from the former
Accelerated Python tree and move any former `tutorials/pyhpc/notebooks` work
into the matching `fundamentals`, `kernels`, `distributed`, or `applications`
directory under `tutorials/accelerated-python/notebooks` in the new deployment.
Do not run `docker compose down --volumes` against the old deployment until
that work has been backed up and verified.

## Notebooks

### Fundamentals

| # | Exercise | Link | Solution |
|---|----------|------|----------|
| 01 | NumPy Intro: `ndarray` Basics | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/01__numpy_intro__ndarray_basics.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/solutions/01__numpy_intro__ndarray_basics__SOLUTION.ipynb) |
| 02 | NumPy Linear Algebra: SVD Reconstruction | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/02__numpy_linear_algebra__svd_reconstruction.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/solutions/02__numpy_linear_algebra__svd_reconstruction__SOLUTION.ipynb) |
| 03 | NumPy to CuPy: `ndarray` Basics | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/03__numpy_to_cupy__ndarray_basics.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/solutions/03__numpy_to_cupy__ndarray_basics__SOLUTION.ipynb) |
| 04 | NumPy to CuPy: SVD Reconstruction | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/04__numpy_to_cupy__svd_reconstruction.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/solutions/04__numpy_to_cupy__svd_reconstruction__SOLUTION.ipynb) |
| 05 | Memory Spaces: Power Iteration | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/05__memory_spaces__power_iteration.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/solutions/05__memory_spaces__power_iteration__SOLUTION.ipynb) |
| 06 | Asynchrony: Power Iteration | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/06__asynchrony__power_iteration.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/solutions/06__asynchrony__power_iteration__SOLUTION.ipynb) |
| 07 | CUDA Core: Devices, Streams and Memory | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/07__cuda_core__devices_streams_and_memory.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/fundamentals/solutions/07__cuda_core__devices_streams_and_memory__SOLUTION.ipynb) |

### Libraries

| # | Exercise | Link | Solution |
|---|----------|------|----------|
| 20 | cuDF: NYC Parking Violations | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/20__cudf__nyc_parking_violations.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/solutions/20__cudf__nyc_parking_violations__SOLUTION.ipynb) |
| 21 | cudf.pandas: NYC Parking Violations | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/21__cudf_pandas__nyc_parking_violations.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/solutions/21__cudf_pandas__nyc_parking_violations__SOLUTION.ipynb) |
| 22 | cuML | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/22__cuml.ipynb) | |
| 23 | CUDA CCCL: Customizing Algorithms | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/23__cuda_cccl__customizing_algorithms.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/solutions/23__cuda_cccl__customizing_algorithms__SOLUTION.ipynb) |
| 24 | nvmath-python: Interop | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/24__nvmath_python__interop.ipynb) | |
| 25 | nvmath-python: Kernel Fusion | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/25__nvmath_python__kernel_fusion.ipynb) | |
| 26 | nvmath-python: Stateful APIs | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/26__nvmath_python__stateful_apis.ipynb) | |
| 27 | nvmath-python: Scaling | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/27__nvmath_python__scaling.ipynb) | |
| 28 | PyNVML | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/libraries/28__pynvml.ipynb) | |

### Kernels

| # | Exercise | Link | Solution |
|---|----------|------|----------|
| 40 | Kernel Authoring: Copy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/40__kernel_authoring__copy.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/solutions/40__kernel_authoring__copy__SOLUTION.ipynb) |
| 41 | Kernel Authoring: Book Histogram | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/41__kernel_authoring__book_histogram.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/solutions/41__kernel_authoring__book_histogram__SOLUTION.ipynb) |
| 42 | Kernel Authoring: Gaussian Blur | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/42__kernel_authoring__gaussian_blur.ipynb) | |
| 43 | Kernel Authoring: Black and White | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/43__kernel_authoring__black_and_white.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/solutions/43__kernel_authoring__black_and_white__SOLUTION.ipynb) |
| 44 | cuTile Python Intro: Vector Add | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/44__cutile_python_intro__vector_add.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/solutions/44__cutile_python_intro__vector_add__SOLUTION.ipynb) |
| 45 | cuTile Python Tiles: Matrix Add | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/45__cutile_python__matrix_add.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/solutions/45__cutile_python__matrix_add__SOLUTION.ipynb) |
| 46 | cuTile Python: Transpose | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/46__cutile_python__transpose.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/solutions/46__cutile_python__transpose__SOLUTION.ipynb) |
| 47 | cuTile Python: Activation Functions | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/47__cutile_python__activation_functions.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/kernels/solutions/47__cutile_python__activation_functions__SOLUTION.ipynb) |

### Distributed

| # | Exercise | Link | Solution |
|---|----------|------|----------|
| 60 | mpi4py | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/distributed/60__mpi4py.ipynb) | |
| 61 | Dask | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/distributed/61__dask.ipynb) | |
| 62 | mpi4py: Heat Equation | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/distributed/62__mpi4py__heat_equation.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/distributed/solutions/62__mpi4py__heat_equation__SOLUTION.ipynb) |

### Applications

| # | Exercise | Link | Solution |
|---|----------|------|----------|
| 80 | C++ Interoperability | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/80__cpp_interop.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/solutions/80__cpp_interop__SOLUTION.ipynb) |
| 81 | Shallow Water Equations: NumPy Baseline | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/81__swe__intro.ipynb) | |
| 82 | Shallow Water Equations: JAX | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/82__swe__jax.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/solutions/82__swe__jax__SOLUTION.ipynb) |
| 83 | Shallow Water Equations: PyOMP | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/83__swe__pyomp.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/solutions/83__swe__pyomp__SOLUTION.ipynb) |
| 84 | Shallow Water Equations: nanobind | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/84__swe__nanobind.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/solutions/84__swe__nanobind__SOLUTION.ipynb) |
| 85 | Shallow Water Equations: CppJIT and CUB | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/85__swe__cppjit__cub.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/solutions/85__swe__cppjit__cub__SOLUTION.ipynb) |
| 86 | Shallow Water Equations: mpi4py | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/86__swe__mpi4py.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/solutions/86__swe__mpi4py__SOLUTION.ipynb) |
| 87 | Shallow Water Equations: Synthesis | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/applications/87__swe__synthesis.ipynb) | |
