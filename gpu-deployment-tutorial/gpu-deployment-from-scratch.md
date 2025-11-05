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

The second half of the tutorial will discuss how to get your own GPUs on the cloud in more general terms. In order to dig into some of the things we will be learning we will be launching a VM through the [NVIDIA Brev](https://brev.nvidia.com) portal. 

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

- Install the `brev` CLI
  - `brew install brevdev/homebrew-brev/brev && brev login`
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

Next we need the core CUDA libraries in order to run any CUDA code. Often these will be installed at the system level in `/usr/local/cuda`.

```bash
ls -ld /usr/local/cuda*
```

If these are missing we need to decide how to get those dependencies. The way we do this is different depending on whether we want to use `pip`/`uv` or `conda`/`pixi` for our Python package manager.

### Python Software environments

#### Pip

If we want to install our packages with `pip` we need to install the CUDA core libraries at the system level, we can do this on Ubuntu with `apt`.

```bash
# Add the NVIDIA repos
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install all of CUDA Toolkit (you need to specify the CUDA version that matches your driver)
sudo apt-get -y install cuda-toolkit-12-8
```

Now that we have our CUDA libraries we can install Python libraries with corresponding versions.

> [!IMPORTANT]
> We need to include the CUDA version in the package name due to limitations in the Python packaging spec, see the [wheelnext](https://wheelnext.dev/) project for plans to solve this in the long term. There is an experimental build of [uv](https://astral.sh/blog/wheel-variants) that supports wheel variants today.

> [!NOTE]
> We also need to use a custom index because the RAPIDS packages tend to be too large for uploading to PyPI. While we can work with them to increase those limits we can run our own index and handle the cost of serving those packages.
> The reason CUDA packages are so large is because GPU machine code varies between models in a way that doesn't happen with CPUs. To work around this CUDA builds for all common GPUs and bundles them together. Further improvements in packaging could help with this in the future.

NOTE: we don't need extra index any more
maybe try cuda 13?

```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12

python  # Start Python interpreter
```

Then we can import `cudf` and allocate some GPU memory

```python
import cudf
s = cudf.Series([1, 2, 3, None, 4])
```

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
> - We don't need CUDA toolkit because each individual CUDA library is available as a conda package. So cudf can depend on them directly and install the ones it needs.
> - Conda supports [virtual packages](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-virtual.html) which allow the solver to discover additional information about the system such as the CUDA version and then pull in the correct package build for your system.

Then we can import `cudf` and allocate some GPU memory

```python
import cudf
s = cudf.Series([1, 2, 3, None, 4])
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

#### cudf.pandas profilers
Some tools and libraries have built in profiling tools. For example the [cudf.pandas](https://github.com/rapidsai-community/tutorial/blob/main/2.cudf_pandas.ipynb) plugin allows you to profile your code from withing Jupyter.

```python
%%cudf.pandas.profile

small_df = pd.DataFrame({"a": ["0", "1", "2"], "b": ["x", "y", "z"]})
small_df = pd.concat([small_df, small_df])

axis = 0
for i in range(0, 2):
    small_df.min(axis=axis)
    axis = i

counts = small_df.groupby("a").b.count()
```

Further reading:
- [cudf.pandas documentation](https://docs.rapids.ai/api/cudf/latest/cudf_pandas/usage/#profiling-cudf-pandas~)

#### NSight Systems and nsys

NVIDIA produces debugging tools which allow you to view low level traces from the GPU kernel execution to find performance bottlenecks.

Typically Python users will run their code with `nsys` to produce a report, and then open it in Nsight as a local viewer.

Like many debugging tools we need to use `nsys` to call Python initially. This will run your code and then output a tracefile which you can download and explore locally.

```bash
nsys profile \
  --trace cuda,osrt,nvtx \
  --gpu-metrics-device=all \
  --cuda-memory-usage true \
  --force-overwrite true \
  --output profile_run_v1 \
  python your_script.py
# Will create profile_your_script.nsys-rep
```

If you are running Jupyter and NSight on the same machine you can also use the [Jupyter Lab Nsight extension](https://pypi.org/project/jupyterlab-nvidia-nsight/)

Further reading:
- [Nsight Documentation](https://developer.nvidia.com/nsight-systems/get-started)
- [Towards Data Science community guide](https://medium.com/data-science/profiling-cuda-using-nsight-systems-a-numba-example-fc65003f8c52)

### How do I do all this on "foo" platform?
Now that we've experimented with all of these tools, libraries and debuggers on a Ubuntu VM the next thing most folks need to figure out is how to apply this to your world. It's likely that you have some opinionated set of hardware/software/platform that you need to use. Perhaps your employer provides you with access to [Databricks](https://docs.rapids.ai/deployment/stable/platforms/databricks/), [Coiled](https://docs.rapids.ai/deployment/stable/platforms/coiled/) or [Snowflake](https://docs.rapids.ai/deployment/stable/platforms/snowflake/). Or maybe you have cloud access and you use services such as [AWS SageMaker](https://docs.rapids.ai/deployment/stable/cloud/aws/sagemaker/), [Azure Machine Learning](https://docs.rapids.ai/deployment/stable/cloud/azure/azureml/) or [Google Cloud Vertex AI](https://docs.rapids.ai/deployment/stable/cloud/gcp/vertex-ai/). Or maybe you have an existing machine or cluster somewhere.

However you get access to GPUs it inevitably falls to you to close the gap between the software provided and the software you need. In our Brev example we got Ubuntu with the NVIDIA driver, but nothing else. On platforms like Snowflake you will get some version of CUDA Toolkit and a few libraries out of the box, but you'll need to figure out how to add the additional things you need.

In [RAPIDS we endeavour to document the most commonly used platforms](https://docs.rapids.ai/deployment/stable/) and how to get from their out of the box offering to a fully functional RAPIDS environment. 

If you're using something that we haven't documented then you can walk through the various levels we've covered and figure out what you have, and what you need and hopefully you now have the ability to get started anywhere. If you think you're using a platform that we should document then [open an issue](https://github.com/rapidsai/deployment/issues/new).`