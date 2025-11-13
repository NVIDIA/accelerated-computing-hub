![nvmath-python](images/nvmath_head_panel@0.25x.png)

# nvmath-python Tutorial

This repository contains a series of tutorial notebooks demonstrating the capabilities and features of **nvmath-python**, NVIDIA's library that bridges Python's scientific computing ecosystem with NVIDIA's CUDA-X math libraries.

## Overview

nvmath-python is designed to provide high-performance mathematical operations that complement existing GPU libraries like CuPy and PyTorch. Unlike traditional array libraries, nvmath-python focuses on advanced mathematical operations with features like kernel fusion, flexible APIs, and device-level integration.

These notebooks can be run on [NVIDIA Brev](https://brev.nvidia.com) or [Google Colab](https://colab.research.google.com).

There are [Docker Images](https://github.com/NVIDIA/accelerated-computing-hub/pkgs/container/nvmath-python-tutorial) and [Docker Compose files](./brev/docker-compose.yml) for creating Brev Launchables or running locally.

Brev Launchables of this tutorial should use:
- 4xL4, 2xL4, 2xL40S, or 1x L40S instances.
- Crusoe or any other provider with Flexible Ports.

## Prerequisites

To use these notebooks, you will need:
- A computer equipped with an NVIDIA GPU
- Python environment with required libraries installed

Please refer to the [nvmath-python documentation](https://docs.nvidia.com/cuda/nvmath-python/0.2.1/installation.html#install-nvmath-python) for installation instructions.

## Notebooks

| Notebook | Tutorial | Solutions |
|----------|----------|-----------|
| 01. Kernel Fusion | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/01_kernel_fusion.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/01_kernel_fusion_SOLUTION.ipynb) |
| 02. Memory and Execution Spaces | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/02_mem_exec_spaces.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/02_mem_exec_spaces_SOLUTION.ipynb) |
| 03. Stateful API and Autotuning | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/03_stateful_api.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/03_stateful_api_SOLUTION.ipynb) |
| 04. FFT Callbacks | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/04_callbacks.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/04_callbacks_SOLUTION.ipynb) |
| 05. Device API | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/05_device_api.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/05_device_api_SOLUTION.ipynb) |
| 06. Direct Sparse Solver | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/06_sparse_solver.ipynb) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/nvmath-python/notebooks/06_sparse_solver_SOLUTION.ipynb) |

---

### 01. Kernel Fusion ([`01_kernel_fusion.ipynb`](notebooks/01_kernel_fusion.ipynb))

**Key Topics:**
- Introduction to nvmath-python and its role in the Python scientific computing ecosystem
- Why nvmath-python is NOT an array library but co-exists with NumPy, CuPy, and PyTorch
- Benchmarking GPU codes with `cupyx.profiler.benchmark`
- Kernel fusion for composite operations (GEMM: `D = α(A·B) + βC`)
- Performance comparison: kernel fusion vs. separate operations
- Using NVIDIA Nsight plugin for JupyterLab for performance profiling

**Key Insights:**
- nvmath-python provides performance benefits over CuPy for GEMM operations through kernel fusion
- Fused kernels eliminate multiple kernel invocation overhead and optimize memory accesses
- Profiling reveals that CuPy requires multiple kernel launches while nvmath-python uses a single fused kernel

---

### 02. Memory and Execution Spaces ([`02_mem_exec_spaces.ipynb`](notebooks/02_mem_exec_spaces.ipynb))

**Key Topics:**
- Understanding memory and execution spaces in nvmath-python
- Flexibility of working with both CPU and GPU memory
- Difference between specialized APIs and generic APIs
- Performance implications of data transfers between memory spaces
- Using nvmath-python's logging mechanism to understand internal operations

**Key Insights:**
- Memory space (where data is stored) and execution space (where computation happens) may differ
- Specialized APIs like `matmul` only support GPU execution, triggering expensive automatic data transfers for CPU inputs
- Generic APIs like FFT adapt to input location, executing on CPU for CPU inputs and GPU for GPU inputs
- Logging provides visibility into specification, planning, and execution phases

---

### 03. Stateful API and Autotuning ([`03_stateful_api.ipynb`](notebooks/03_stateful_api.ipynb))

**Key Topics:**
- Stateless (function-form) vs. stateful (class-form) APIs
- Understanding the four phases: specification, planning, execution, and resource management
- Reusing `Matmul` objects for batched operations
- Performance benefits of amortizing specification and planning costs
- Autotuning for optimal kernel selection

**Key Insights:**
- Stateless API is convenient for single operations but repeats specification/planning for each call
- Stateful API allows specification and planning once, then multiple executions
- Autotuning finds optimal kernels when built-in heuristics are suboptimal
- Critical for scenarios with repeated operations on similar-shaped data

---

### 04. FFT Callbacks ([`04_callbacks.ipynb`](notebooks/04_callbacks.ipynb))

**Key Topics:**
- Custom Python functions as FFT prolog/epilog callbacks
- JIT compilation to intermediate representation (LTO-IR)
- Application example: Gaussian image filtering using FFT
- Comparison of CuPy vs. nvmath-python implementations
- Cost breakdown: compilation, planning, and execution phases
- Amortizing compilation/planning costs across batch processing

**Key Insights:**
- Callbacks enable custom element-wise operations fused with FFT kernels
- JIT compilation overhead is one-time cost that can be amortized
- For single images, CuPy may be faster due to nvmath-python's compilation overhead
- For large enough batches amortization makes stateful API a preferred choice

---

### 05. Device API ([`05_device_api.ipynb`](notebooks/05_device_api.ipynb))

**Key Topics:**
- Using nvmath-python's device APIs within custom numba-cuda kernels
- Monte Carlo simulation of stock prices using Geometric Brownian Motion (GBM)
- Integration with nvmath-python's random number generation at device level
- Comparison of memory-bound array operations vs. compute-intensive kernels
- Optimizing throughput by consuming multiple random variates per iteration

**Key Insights:**
- Custom kernels eliminate intermediate array allocations and memory transfers
- nvmath-python device RNG provides performance benefits over CuPy for GBM simulation
- Philox4_32_10 generator returns 4 random variates at once, enabling vectorized consumption
- Device-level APIs enable fine-grained control for compute-intensive workloads
- Critical for applications where each thread handles complex calculations

---

### 06. Direct Sparse Solver ([`06_sparse_solver.ipynb`](notebooks/06_sparse_solver.ipynb))

**Key Topics:**
- Direct sparse solver for large linear systems with sparse matrices
- Solving linear equations of the form A·X = B using nvmath-python
- Working with CSR (Compressed Sparse Row) format matrices
- GPU and hybrid execution modes for the solver

**Key Insights:**
- nvmath-python provides high-performance sparse solver backed by NVIDIA cuDSS library
- Direct methods are suitable for sparse linear systems with specific matrix structures
- CSR format efficiently stores and manipulates sparse matrices on GPU

---

## General Benchmarking Notes

All notebooks use a consistent benchmarking approach with `cupyx.profiler.benchmark`:
- Proper GPU synchronization using CUDA events
- Warm-up runs to eliminate cold-start effects
- Multiple repetitions for statistical stability
- Reports minimum time from repeated runs

For detailed profiling, notebooks demonstrate using NVIDIA Nsight Systems and the JupyterLab plugin for kernel-level analysis.

## Additional Resources

- [nvmath-python Documentation](https://docs.nvidia.com/cuda/nvmath-python/)
- [NVIDIA CUDA-X Libraries](https://developer.nvidia.com/gpu-accelerated-libraries)
- [NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-systems)
