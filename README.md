# NVIDIA Accelerated Computing Hub

This repository is a home for open learning materials related to GPU computing.  You will find user guides, tutorials, and other works freely available for all learners interested in GPU computing.

The intention is to create a living project where documents, examples, best practices, optimizations, and new features can become both visible and accessible to users quickly and easily.

## Tutorials and Syllabi

The following interactive tutorials are available and can be used on [NVIDIA Brev](https://brev.nvidia.com) or [Google Colab](https://colab.research.google.com). See [the documentation](docs/brev.md) for more information on creating and deploying Brev Launchables using this content.

| Content | Docker Compose | Brev Instance | Brev Provider |
|---------|----------------|---------------|---------------|
| [CUDA Tile Tutorial](tutorials/cuda-tile/README.md) | [docker-compose.yml](tutorials/cuda-tile/brev/docker-compose.yml) | 1xB300, 1xB200, or 1xRTXPro6000 | Any Blackwell provider; none have Flexible Ports yet |
| [CUDA C++ Tutorial](tutorials/cuda-cpp/README.md) | [docker-compose.yml](tutorials/cuda-cpp/brev/docker-compose.yml) | L40S, L4, or T4 | Crusoe or any other with Flexible Ports |
| [Standard Parallelism Tutorial](tutorials/stdpar/README.md) | [docker-compose.yml](tutorials/stdpar/brev/docker-compose.yml) | 4xL4, 2xL4, 2xL40S, or 1x L40S | GCP, AWS, or any other with Flexible Ports and Linux 6.1.24+, 6.2.11+, or 6.3+ (for HMM) |
| [Accelerated Python Tutorial](tutorials/accelerated-python/README.md) | [docker-compose.yml](tutorials/accelerated-python/brev/docker-compose.yml) | L40S, L4, or T4; 4xL4 or 2xL4 for distributed | Crusoe or any other with Flexible Ports |
| [nvmath-python Tutorial](tutorials/nvmath-python/README.md) | [docker-compose.yml](tutorials/nvmath-python/brev/docker-compose.yml) | 4xL4, 2xL4, 2xL40S, or 1x L40S | Crusoe or any other with Flexible Ports |
| [CUDA Python Tutorial - CuPy, cuDF, CCCL, & Kernels - 8 Hours](tutorials/accelerated-python/notebooks/syllabi/cuda_python__cupy_cudf_cccl_kernels__8_hours.ipynb) | [docker-compose.yml](https://github.com/NVIDIA/accelerated-computing-hub/blob/generated/main/tutorials/accelerated-python/notebooks/syllabi/cuda_python__cupy_cudf_cccl_kernels__8_hours__docker_compose.yml) | L40S, L4, or T4 | Crusoe or any other with Flexible Ports |
| [CUDA Python Tutorial - cuda.core & CCCL - 2 Hours](tutorials/accelerated-python/notebooks/syllabi/cuda_python__cuda_core_cccl__2_hours.ipynb) | [docker-compose.yml](https://github.com/NVIDIA/accelerated-computing-hub/blob/generated/main/tutorials/accelerated-python/notebooks/syllabi/cuda_python__cuda_core_cccl__2_hours__docker_compose.yml) | L40S, L4, or T4 | Crusoe or any other with Flexible Ports |
| [PyHPC Tutorial - NumPy, CuPy, & mpi4py - 4 Hours](tutorials/accelerated-python/notebooks/syllabi/pyhpc__numpy_cupy_mpi4py__4_hours.ipynb) | [docker-compose.yml](https://github.com/NVIDIA/accelerated-computing-hub/blob/generated/main/tutorials/accelerated-python/notebooks/syllabi/pyhpc__numpy_cupy_mpi4py__4_hours__docker_compose.yml) | 4xL4, 2xL4, 2xL40S, or 1x L40S | Crusoe or any other with Flexible Ports |
| [GPU Deployment Tutorial](tutorials/gpu-deployment/gpu-deployment-from-scratch.md) | | |

## License

All written materials (user guides, documentation, presentations) are subject to [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

All codes (notebook code, coding examples) are subject to [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Contact

For additional help, please use [NVIDIA's CUDA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda).
