# GPU development with Python 101 Tutorial

**Welcome to the GPU Development in Python 101 tutorial.**

Since joining NVIDIA I’ve gotten to grips with the fundamentals of writing accelerated code in Python. I was amazed to discover that I didn’t need to learn C++ and I didn’t need new development tools. Writing GPU code in Python is easier today than ever, and in this tutorial, I will share what I’ve learned and how you can get started with accelerating your code.

In this tutorial we will cover:
- What is a GPU and why is it different to a CPU?
- An overview of the CUDA development model.
- Numba: A high performance compiler for Python.
- Writing your first GPU code in Python.
- Managing memory.
- Understanding what your GPU is doing with pyNVML (memory usage, utilization, etc).
- RAPIDS: A suite of GPU accelerated data science libraries.
- Working with Numpy style arrays on the GPU.
- Working with Pandas style dataframes on the GPU.
- Performing some scikit-learn style machine learning on the GPU.

Attendees will be expected to have a general knowledge of Python and programming concepts, but no GPU experience will be necessary. The key takeaway for attendees will be the knowledge that they don’t have to do much differently to get their code running on a GPU.

## Running the tutorial

This tutorial has been updated to run on [Google Colab](https://colab.google/) (as of Dec 2023) which allows anyone with a Google account to get an interactive Python environment with a GPU.

To run each notebook you need to click the link on the table below to open it in Colab, and then set the runtime to include a GPU. **You will need to do this for every notebook.**

To do this you need to:
- Select the arrow next to "Connect" in the top right
- Select "Change runtime type"
- Choose a GPU (the T4 is available in the free tier and is more than enough for this tutorial)
- Hit save

![](images/colab-runtime-type.png)

![](images/colab-t4.png)

## Notebooks 

| Notebook      | Link |
| ----------- | ----------- |
| 0.0 Welcome | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/0.0_Welcome.ipynb)|
| 1.0 CPU GPU Comparison | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/1.0_CPU_GPU_Comparison.ipynb)|
| 2.0 Numba | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/2.0_20Numba.ipynb)|
| 2.1 Numba Lab | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/2.1_Numba_lab.ipynb)|
| 2.2 Numba Lab (solution) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/2.2_Numba_lab_(solution).ipynb)|
| 3.0 Numba Gauss | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/3.0_Numba_gauss.ipynb)|
| 3.1 Numba Lab 2 | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/3.1_Numba_lab_2.ipynb)|
| 3.2 Numba Lab 2 (solution) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/3.2_Numba_lab_2_(solution).ipynb)|
| 4.0 pyNVML | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/4.0_pyNVML.ipynb)|
| 4.1 CUDA Array Interface | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/4.1_CUDA_Array_Interface.ipynb)|
| 5.0 Cupy | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/5.0_Cupy.ipynb)|
| 5.1 Cupy Lab | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/5.1_Cupy_Lab.ipynb)|
| 5.2 Cupy Lab (solution) | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/5.2_Cupy_Lab_(solution).ipynb)|
| 6.0 cuDF | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/6.0_cuDF.ipynb)|
| 7.0 cuML | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/7.0_cuML.ipynb)|
| 8.0 Multi-GPU with Dask | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jnbntz/accelerated-computing-hub/blob/main/gpu-python-tutorial/8.0_Multi-GPU_with_Dask.ipynb)|
