# Accelerated Python Tutorial

This modular tutorial contains content on all things related to accelerated Python:

- [Notebooks](./notebooks) containing lessons and exercises, intended for self-paced or instructor-led learning, which can be run on [NVIDIA Brev](https://brev.nvidia.com) or [Google Colab](https://colab.research.google.com).
- [Slides](./slides) containing the lecture content for the lessons.
- [Syllabi](./notebooks/syllabi) that select a subset of the notebooks for a particular learning objective.
- [Docker Images](https://github.com/NVIDIA/accelerated-computing-hub/pkgs/container/accelerated-python-tutorial) and [Docker Compose files](./brev/docker-compose.yml) for creating Brev Launchables or running locally.

Brev Launchables of this tutorial should use:
- L40S, L4, or T4 instances (for non-distributed notebooks).
- 4xL4 or 2xL4 instances (for distributed notebooks).
- Crusoe or any other provider with Flexible Ports.

## Syllabi

- [CUDA Python - CuPy, cuDF, CCCL, & Kernels - 8 Hours](./notebooks/syllabi/cuda_python__cupy_cudf_cccl_kernels__8_hours.ipynb).
- [CUDA Python - cuda.core & CCCL - 2 Hours](./notebooks/syllabi/cuda_python__cuda_core_cccl__2_hours.ipynb)
- [PyHPC - NumPy, CuPy, & mpi4py - 4 Hours](./notebooks/syllabi/pyhpc__numpy_cupy_mpi4py__4_hours.ipynb)

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

### Distributed

| # | Exercise | Link | Solution |
|---|----------|------|----------|
| 60 | mpi4py | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/distributed/60__mpi4py.ipynb) | |
| 61 | Dask | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/accelerated-python/notebooks/distributed/61__dask.ipynb) | |
