# Fundamentals of Accelerated Computing with Modern CUDA C++

This tutorial teaches you the fundamentals of GPU programming and modern CUDA C++. You can watch
lectures corresponding to this course on [YouTube](https://www.youtube.com/playlist?list=PL5B692fm6--vWLhYPqLcEu6RF3hXjEyJr).
You'll find the following content:

- [Notebooks](./notebooks) containing lessons and exercises, intended for self-paced or instructor-led learning, which can be run on [NVIDIA Brev](https://brev.nvidia.com) or [Google Colab](https://colab.research.google.com).
- [Slides](./slides) containing the lecture content for the lessons.
- [Docker Images](https://github.com/NVIDIA/accelerated-computing-hub/pkgs/container/cuda-cpp-tutorial) and [Docker Compose files](./brev/docker-compose.yml) for creating Brev Launchables or running locally.

Brev Launchables of this tutorial should use:
- L40S, L4, or T4 instances.
- Crusoe or any other provider with Flexible Ports.

## Notebooks

### CUDA Made Easy: Accelerating Applications with Parallel Algorithms

| Notebook                                | Link                                                                                                                                                                                       |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 01.01.01 Introduction | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.01-Introduction/01.01.01-CUDA-Made-Easy.ipynb) |
| 01.02.01 Execution Spaces               | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.02-Execution-Spaces/01.02.01-Execution-Spaces.ipynb) |
| 01.02.02 Exercise Annotate Execution Spaces   | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.02-Execution-Spaces/01.02.02-Exercise-Annotate-Execution-Spaces.ipynb) |
| 01.02.03 Exercise Changing Execution Space    | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.02-Execution-Spaces/01.02.03-Exercise-Changing-Execution-Space.ipynb) |
| 01.02.04 Exercise Compute Median Temperature   | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.02-Execution-Spaces/01.02.04-Exercise-Compute-Median-Temperature.ipynb) |
| 01.03.01 Extending Algorithms           | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.03-Extending-Algorithms/01.03.01-Extending-Algorithms.ipynb) |
| 01.03.02 Exercise Computing Variance     | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.03-Extending-Algorithms/01.03.02-Exercise-Computing-Variance.ipynb) |
| 01.04.01 Vocabulary Types              | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.04-Vocabulary-Types/01.04.01-Vocabulary-Types.ipynb) |
| 01.04.02 Exercise Mdspan                | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.04-Vocabulary-Types/01.04.02-Exercise-mdspan.ipynb) |
| 01.05.01 Serial vs Parallel            | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.05-Serial-vs-Parallel/01.05.01-Serial-vs-Parallel.ipynb) |
| 01.05.02 Exercise Segmented Sum Optimization   | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.05-Serial-vs-Parallel/01.05.02-Exercise-Segmented-Sum-Optimization.ipynb) |
| 01.05.03 Exercise Segmented Mean        | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.05-Serial-vs-Parallel/01.05.03-Exercise-Segmented-Mean.ipynb) |
| 01.06.01 Memory Spaces                 | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.06-Memory-Spaces/01.06.01-Memory-Spaces.ipynb) |
| 01.06.02 Exercise Copy                  | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.06-Memory-Spaces/01.06.02-Exercise-Copy.ipynb) |
| 01.07.01 Summary                       | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.07-Summary/01.07.01-Summary.ipynb) |
| 01.08.01 Advanced                      | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/01.08-Advanced/01.08.01-Advanced.ipynb) |

### Unlocking the GPU’s Full Potential: Asynchrony and CUDA Streams

| Notebook                                | Link                                                                                                                                                                                       |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 02.01.01 Introduction                  | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.01-Introduction/02.01.01-Introduction.ipynb) |
| 02.02.01 Asynchrony                    | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.02-Asynchrony/02.02.01-Asynchrony.ipynb) |
| 02.02.02 Exercise Compute IO Overlap    | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.02-Asynchrony/02.02.02-Exercise-Compute-IO-Overlap.ipynb) |
| 02.02.03 Exercise Nsight               | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.02-Asynchrony/02.02.03-Exercise-Nsight.ipynb) |
| 02.02.04 Exercise NVTX                 | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.02-Asynchrony/02.02.04-Exercise-NVTX.ipynb) |
| 02.03.01 Streams                       | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.03-Streams/02.03.01-Streams.ipynb) |
| 02.03.02 Exercise Async Copy            | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.03-Streams/02.03.02-Exercise-Async-Copy.ipynb) |
| 02.04.01 Pinned                        | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.04-Pinned-Memory/02.04.01-Pinned.ipynb) |
| 02.04.02 Exercise Copy Overlap          | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/02.04-Pinned-Memory/02.04.02-Exercise-Copy-Overlap.ipynb) |

### Implementing New Algorithms with CUDA Kernels

| Notebook                                | Link                                                                                                                                                                                       |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 03.01.01 Introduction                     | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.01-Introduction/03.01-Introduction.ipynb) |
| 03.02.01 Kernels                       | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.02-Kernels/03.02.01-Kernels.ipynb) |
| 03.02.02 Exercise Symmetry             | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.02-Kernels/03.02.02-Exercise-Symmetry.ipynb) |
| 03.02.03 Exercise Row Symmetry          | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.02-Kernels/03.02.03-Exercise-Row-Symmetry.ipynb) |
| 03.02.04 Dev Tools                     | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.02-Kernels/03.02.04-Dev-Tools.ipynb) |
| 03.03.01 Histogram                     | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.03-Atomics/03.03.01-Histogram.ipynb) |
| 03.03.02 Exercise Fix Histogram         | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.03-Atomics/03.03.02-Exercise-Fix-Histogram.ipynb) |
| 03.04.01 Sync                          | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.04-Synchronization/03.04.01-Sync.ipynb) |
| 03.04.02 Exercise Histogram             | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.04-Synchronization/03.04.02-Exercise-Histogram.ipynb) |
| 03.05.01 Shared                        | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.05-Shared-Memory/03.05.01-Shared.ipynb) |
| 03.05.02 Exercise Optimize Histogram    | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.05-Shared-Memory/03.05.02-Exercise-Optimize-Histogram.ipynb) |
| 03.06.01 Cooperative                   | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.06-Cooperative-Algorithms/03.06.01-Cooperative.ipynb) |
| 03.06.02 Exercise Cooperative Histogram | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/main/tutorials/cuda-cpp/notebooks/03.06-Cooperative-Algorithms/03.06.02-Exercise-Cooperative-Histogram.ipynb) |
