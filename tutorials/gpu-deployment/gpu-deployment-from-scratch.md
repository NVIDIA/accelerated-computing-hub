# GPU Deployment Guide 
Guide Adapted from:[**EuroSciPy 2025**](https://euroscipy.org/schedule/) | [Share Link](https://jacobtomlinson.dev/links/euroscipy-2025)

## Agenda
- Deployment (40 mins)
  - NVIDIA Brev
  - GPU Software Environment Fundamentals
  - Python packages that use CUDA
  - Monitoring/debugging tools
  - Other platforms

## Deployment

This tutorial will discuss how to get your own GPUs on the cloud in more general terms. In order to dig into some of the 
things we will be learning, we will launch a VM through the [NVIDIA Brev](https://brev.nvidia.com) portal. 

### Getting set up with Brev

- Sign in to or register an account at https://brev.nvidia.com
- Ensure you are a member of an organization
  - One should be created for you when you register, but if not it will say "undefined" in the top right
  - If you don't have one you can create a new one and give it a name
- Apply credits to your organization
  - Navigate to Billing
  - Select "Redeem Code" <UPDATE IN HERE>

<img width="1398" height="594" alt="Screenshot 2025-08-13 at 15 16 09" src="https://gist.github.com/user-attachments/assets/45f9b71a-eb5f-471f-93fa-2503cf8f4714" />

#### Launching a Brev VM

- Under "GPUs" select "New Instance"
- Choose a GPU type that costs <$1/hour (e.g an L4) 
- Choose any provider
- Give your VM a name
- Hit Deploy

#### Connecting to your VM

Once your VM is deployed, follow the Brev access instructions provided for your instance. The connection instructions will vary depending on your operating system. For example, on macOS you would:

- Install the `brev` CLI
  - `brew install brevdev/homebrew-brev/brev 
- Login to your account (copy from access page)
  - `brev login --token ****`
- Connect via SSH
  - `brev shell <your vm name>`

For Linux and Windows instructions check the [brev-cli install documentation](https://docs.nvidia.com/brev/latest/brev-cli.html#installation-instructions)

### Exploring our GPU Software Environment

Let's start by exploring our VM to see what software we got out of the box.

```console
$ cat /etc/os-release
PRETTY_NAME="Ubuntu 22.04.5 LTS"
```

We can check our GPU information by running `nvidia-smi`.

```console
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.158.01             Driver Version: 570.158.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L4                      On  |   00000000:00:03.0 Off |                    0 |
| N/A   47C    P8             13W /   72W |       0MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

### Let's install something

```console
$which pip
/usr/bin/pip
```

If you have pip, we can try to install something like `cupy`

```console
pip install cupy-cuda12x
```

```console
$python3
>>> import cupy as cp
>>> x_gpu = cp.array([1, 2, 3])
>>> x2 = x_gpu**2
Traceback (most recent call last):
  File "cupy_backends/cuda/_softlink.pyx", line 25, in cupy_backends.cuda._softlink.SoftLink.__init__
  File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libnvrtc.so.12: cannot open shared object file: No such file or directory

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "cupy/_core/core.pyx", line 1448, in cupy._core.core._ndarray_base.__pow__
  File "cupy/_core/core.pyx", line 1799, in cupy._core.core._ndarray_base.__array_ufunc__
  File "cupy/_core/_kernel.pyx", line 1374, in cupy._core._kernel.ufunc.__call__
  File "cupy/_core/_kernel.pyx", line 1401, in cupy._core._kernel.ufunc._get_ufunc_kernel
  File "cupy/_core/_kernel.pyx", line 1082, in cupy._core._kernel._get_ufunc_kernel
  File "cupy/_core/_kernel.pyx", line 94, in cupy._core._kernel._get_simple_elementwise_kernel
  File "cupy/_core/_kernel.pyx", line 82, in cupy._core._kernel._get_simple_elementwise_kernel_from_code
  File "cupy/_core/core.pyx", line 2375, in cupy._core.core.compile_with_cache
  File "cupy/_core/core.pyx", line 2320, in cupy._core.core.assemble_cupy_compiler_options
  File "cupy_backends/cuda/libs/nvrtc.pyx", line 57, in cupy_backends.cuda.libs.nvrtc.getVersion
  File "cupy_backends/cuda/libs/_cnvrtc.pxi", line 72, in cupy_backends.cuda.libs.nvrtc.initialize
  File "cupy_backends/cuda/libs/_cnvrtc.pxi", line 75, in cupy_backends.cuda.libs.nvrtc._initialize
  File "cupy_backends/cuda/libs/_cnvrtc.pxi", line 153, in cupy_backends.cuda.libs.nvrtc._get_softlink
  File "cupy_backends/cuda/_softlink.pyx", line 32, in cupy_backends.cuda._softlink.SoftLink.__init__
RuntimeError: CuPy failed to load libnvrtc.so.12: OSError: libnvrtc.so.12: cannot open shared object file: No such file or directory
```

**What does this error mean?**

This error indicates that CuPy cannot find the CUDA runtime libraries it needs to work.  It's looking for `libnvrtc.so.12` (the
NVIDIA Runtime Compiler library for CUDA 12), but it's not installed or not in the system's library path.

NVRTC is used to JIT (just-in-time) compile CUDA code at runtime. When we run `x_gpu**2`, CuPy needs NVRTC to dynamically compile a
GPU kernel for this operation.

We need the core CUDA libraries in order to run any CUDA code. Often these will be installed at the system level
in `/usr/local/cuda`. Let's check that:

```bash
ls -ld /usr/local/cuda*
```

If these are missing we need to decide how to get those dependencies. The way we do this is different depending on whether we want to use `pip`/`uv` or `conda`/`pixi` for our Python package manager.


### Python Software environments

At the moment, when you install CuPy with pip, the package has dependencies on CUDA libraries that aren't available on PyPI. CuPy expects to find these CUDA libraries already installed on your system at `/usr/local/cuda` or in the system library path. This is why we need to install the CUDA Toolkit separately using the system package manager.

> [!NOTE]
> For `cupy` this will change in the upcoming release. For `cudf` and `cuml`this is not an issue. Here we are illustrating how to troubleshoot in case you run into this type of errors.

#### Pip

If we want to install our packages with `pip` we need to install the CUDA core libraries at the system level to be safe. We can do this on Ubuntu with `apt`.

Make sure to select the appropriate package that matches your system architecture (x86_64, ARM64, etc.) and
your specific OS distribution and version. The example below shows the installation for Ubuntu 22.04 on
`x86_64` (what we have in our brev instance). For other distributions and architectures, consult the [NVIDIA
CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#ubuntu).

```bash
# Add the NVIDIA repos
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install the CUDA Toolkit (specify the CUDA version that matches your driver - check nvidia-smi)
sudo apt-get -y install cuda-toolkit-12-8
```

Now if we try to run our snippet of code again, we see:

```python
>>> import cupy as cp
>>> x_gpu = cp.array([1, 2, 3])
>>> x2 = x_gpu**2
>>> x2
array([1, 4, 9])
```

**Installing more packages**

Now that we have our CUDA libraries we can install Python libraries with corresponding versions.

> [!IMPORTANT]
> We need to include the CUDA version in the package name due to limitations in the Python packaging spec, see the [wheelnext](https://wheelnext.dev/) project for plans to solve this in the long term. There is an experimental build of [uv](https://astral.sh/blog/wheel-variants) that supports wheel variants today.

> [!NOTE]
> For some packages we need to use a custom index because the RAPIDS packages tend to be too large for uploading to PyPI. While we can work with them to increase those limits we can run our own index and handle the cost of serving those packages. You can check the [RAPIDS installation selector](https://docs.rapids.ai/install/) to see if which package needs the extra index.
> The reason CUDA packages are so large is because GPU machine code varies between models in a way that doesn't happen with CPUs. To work around this CUDA builds for all common GPUs and bundles them together. Further improvements in packaging could help with this in the future.

As of the 25.10 release neither cuDF nor cuML need the extra index. Let's install `cudf` and do a simple operation. 

```bash
pip install cudf-cu12

python3  # Start Python interpreter
```

Then we can import `cudf` and allocate some GPU memory

```python
import cudf
s = cudf.Series([1, 2, 3, None, 4])
s.apply(lambda x: x+1)
```

#### What About uv?

**Installation**

Install `uv` following the [Astral documentation](https://docs.astral.sh/uv/getting-started/installation/#installation-methods):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> [!NOTE]
> You'll need to source your `.bashrc` to make `uv` available in your current shell:

```bash
source ~/.bashrc
```

**Creating a Test Environment**

Let's create a separate directory to experiment with `uv`. We'll set up a Python 3.12 environment and install `cudf`:

```bash
mkdir sandbox
cd sandbox
uv venv --python 3.12
source .venv/bin/activate
uv pip install "cudf-cu12==25.10.*"
```

**Testing the Installation**

Launch the Python interpreter and test with some cuDF code:

```python
import cudf

# Create a cuDF DataFrame
data = {'col1': [1, 2, 3, 4], 'col2': [10, 20, 30, 40]}
df = cudf.DataFrame(data)

# Perform an operation on a DataFrame column
df['col3'] = df['col1'] * df['col2']
df
```

**Important Limitation**

When installing nightly or pre-release versions of packages, `uv` has an all-or-nothing strategy. It requires more explicit configuration when working with nightlies or pre-releases, and failing to do so can generate version conflicts and installation errors that are less common with `pip`. For more information, see the [uv pre-release compatibility documentation](https://docs.astral.sh/uv/pip/compatibility/#pre-release-compatibility).


#### Conda

When installing libraries with conda each individual CUDA library can be installed as a conda package. So we don't need to ensure any of the CUDA libraries already exist in `/usr/local/cuda`.

If you prefer to use `conda` then we need to install it first.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3-$(uname)-$(uname -m).sh  # Follow the prompts
```

Then we can create a new conda environment with `python` and `cudf`.

```bash
conda create -n rapids -c rapidsai -c conda-forge -c nvidia cudf python=3.13 'cuda-version>=12.0,<=12.9'

conda activate rapids
```

> [!NOTE]
> You may notice this is much simpler than the pip installation. This is for two reasons:
> - We don't need to install the CUDA toolkit at the system level because each individual CUDA library is available as a conda package. So cudf can depend on them directly and install the ones it needs.
> - Conda supports [virtual packages](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-virtual.html) which allow the solver to discover additional information about the system such as the CUDA version and then pull in the correct package build for your system.
>
> **Why are these CUDA packages available on conda-forge but not PyPI?**
>
> Historically, Python could only package source distributions (which compile at install time), but NVIDIA doesn't distribute CUDA Toolkit source code. Conda was created as a binary package manager that can package any compiled code. While Python wheels now support binary distributions for pip, they are relatively new and it takes time for the ecosystem to catch up. Conda also provides quality of life improvements like virtual packages (exposing the driver CUDA version to the dependency solver) and optional package constraints, making it currently more mature for complex GPU dependencies.

Then we can import `cudf` and allocate some GPU memory

```python
import cudf
s = cudf.Series(['a', 'aa', 'b'])
s.apply(lambda x: len(x))
```

### Monitoring and debugging tools

When working with GPUs you need to get visibility into what the device is doing. We can get a whole range of information with `nvidia-smi`.

```bash
# Show high level GPU information
nvidia-smi

# List GPUs
nvidia-smi -L

# Dump detailed information
nvidia-smi -q
```

#### NVML

Below `nvidia-smi` sits NVML, a protocol for querying low level information from the GPU. There are Python bindings if you want to access this data yourselv.

```bash
pip install nvidia-ml-py  # Package name doesn't match library name. You import it with `import pynvml`
``` 

Here are some simple examples of using `pynvml`:

```python
import pynvml

# Initialize NVML
pynvml.nvmlInit()

# Get the number of GPUs
num_gpus = pynvml.nvmlDeviceGetCount()
print(f"Number of GPUs: {num_gpus}")

# Get a handle to the first GPU
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

# Get the GPU name
gpu_name = pynvml.nvmlDeviceGetName(gpu)
print(f"GPU Name: {gpu_name}")

# Get memory information (convert bytes to GB)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu)
print(f"Memory Used: {mem_info.used / 1e9:.2f} GB")
print(f"Memory Free: {mem_info.free / 1e9:.2f} GB")
```

You can learn more about using the `pynvml` library in [this notebook on the Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/gpu-python-tutorial/4.0_pyNVML.ipynb).


#### Jupyter Lab NVDashboard
If you are a fan of Jupyter Lab you can view metrics directly in the interface with [jupyterlab-nvdashboard](https://github.com/rapidsai/jupyterlab-nvdashboard).

<img width="1834" height="1010" alt="image" src="https://gist.github.com/user-attachments/assets/463d1a9f-0888-4390-9945-55a6ec58288b" />

> [!NOTE]
> Our Brev VM has Jupyter running for us in the system python environment via systemd. We can install NVDashboard in here but we need to ensure we are installing it into the right Python.

```bash
# Ensure we are using the base Python
conda deactivate  # If you installed conda deactivate it
which python3  # Should be /usr/bin/python3

# Install NVDashboard
pip install jupyterlab_nvdashboard
# alternatively /usr/bin/python3 -m pip install jupyterlab_nvdashboard

# Restart jupyter
sudo systemctl restart jupyter
```

#### nvtop
There also also many great third-party tools out there for inspecting your GPU. One such project is [nvtop](https://github.com/Syllo/nvtop), a CLI tool for viewing GPU stats.

<img width="1402" height="542" alt="image" src="https://gist.github.com/user-attachments/assets/4b8794ca-bdb0-4946-8c82-17eefd751da2" />

```bash
# Install with apt
sudo apt install nvtop

# Start nvtop
nvtop
```

Let's create a simple Python script to keep the GPU busy so we can monitor it with nvtop:

```python
import cupy as cp

arr = cp.arange(1, 50_000_000)

while True: 
    _ = arr**2
```

To monitor this with nvtop:

1. Start running the Python script above
2. Press `Ctrl+Z` to suspend the process
3. Type `bg` to send it to the background
4. Run `nvtop` to monitor GPU activity
5. Observe the GPU memory usage and utilization percentages in nvtop
6. Press `q` to quit nvtop
7. Type `fg` to bring your Python process back to the foreground
8. Press `Ctrl+C` to stop the Python script


#### cudf.pandas profilers
Some tools and libraries have built in profiling tools. For example the [cudf.pandas](https://github.com/rapidsai-community/tutorial/blob/main/2.cudf_pandas.ipynb) plugin allows you to profile your code from withing Jupyter.

```python
%load_ext cudf.pandas
import pandas as pd

%%cudf.pandas.profile

small_df = pd.DataFrame({"a": ["0", "1", "2"], "b": ["x", "y", "z"]})
small_df = pd.concat([small_df, small_df])

small_df.min(axis=0)
small_df.min(axis=1)

counts = small_df.groupby("a").b.count()
```

Further reading:
- [cudf.pandas documentation](https://docs.rapids.ai/api/cudf/latest/cudf_pandas/usage/#profiling-cudf-pandas~)
- [cudf.pandas tutorial - profiling section](https://github.com/rapidsai-community/tutorial/blob/main/2.cudf_pandas.ipynb)

#### NSight Systems and nsys

NVIDIA produces debugging tools which allow you to view low level traces from the GPU kernel execution to find performance bottlenecks.

Typically Python users will run their code with `nsys` to produce a report, and then open it in Nsight as a local viewer.

Like many debugging tools we need to use `nsys` to call Python initially. This will run your code and then output a tracefile which you can download and explore locally.

Let's create script name `my_script.py`

```python
import cudf.pandas
cudf.pandas.install()

import pandas as pd

small_df = pd.DataFrame({"a": ["0", "1", "2"], "b": ["x", "y", "z"]})
small_df = pd.concat([small_df, small_df])

small_df.min(axis=0)
small_df.min(axis=1)

counts = small_df.groupby("a").b.count()
```

Now we run the script with `nsys` 

> [!NOTE]
> If you are using the python from the `uv venv` use the python from the venv, to do that replace python for 
>  `/home/ubuntu/sandbox/.venv/bin/python my_script.py`


```bash
sudo nsys profile \
  --trace cuda,osrt,nvtx \
  --gpu-metrics-device all \
  --cuda-memory-usage true \
  --force-overwrite true \
  --output profile_run_v1 \
  python my_script.py
# Will create profile_my_script.nsys-rep
```

To be able to visualize the file, we can download it an use [nsight-systems](https://developer.nvidia.com/nsight-systems/get-started).

If you are running Jupyter and NSight on the same machine you can also use the [Jupyter Lab Nsight extension](https://pypi.org/project/jupyterlab-nvidia-nsight/)

```bash
pip install jupyterlab-nvidia-nsight
```

Further reading:
- [Nsight Documentation](https://developer.nvidia.com/nsight-systems/get-started)
- [Towards Data Science community guide](https://medium.com/data-science/profiling-cuda-using-nsight-systems-a-numba-example-fc65003f8c52)

### How do I do all this on "foo" platform?
Now that we've experimented with all of these tools, libraries and debuggers on a Ubuntu VM the next thing most folks need to figure out is how to apply this to your world. It's likely that you have some opinionated set of hardware/software/platform that you need to use. Perhaps your employer provides you with access to [Databricks](https://docs.rapids.ai/deployment/stable/platforms/databricks/), [Coiled](https://docs.rapids.ai/deployment/stable/platforms/coiled/) or [Snowflake](https://docs.rapids.ai/deployment/stable/platforms/snowflake/). Or maybe you have cloud access and you use services such as [AWS SageMaker](https://docs.rapids.ai/deployment/stable/cloud/aws/sagemaker/), [Azure Machine Learning](https://docs.rapids.ai/deployment/stable/cloud/azure/azureml/) or [Google Cloud Vertex AI](https://docs.rapids.ai/deployment/stable/cloud/gcp/vertex-ai/). Or maybe you have an existing machine or cluster somewhere.

However you get access to GPUs it inevitably falls to you to close the gap between the software provided and the software you need. In our Brev example we got Ubuntu with the NVIDIA driver, but nothing else. On platforms like Snowflake you will get some version of CUDA Toolkit and a few libraries out of the box, but you'll need to figure out how to add the additional things you need.

In [RAPIDS we endeavour to document the most commonly used platforms](https://docs.rapids.ai/deployment/stable/) and how to get from their out of the box offering to a fully functional RAPIDS environment. 

If you're using something that we haven't documented then you can walk through the various levels we've covered and figure out what you have, and what you need and hopefully you now have the ability to get started anywhere. If you think you're using a platform that we should document then [open an issue](https://github.com/rapidsai/deployment/issues/new).`