"""
HPCCM development container for the stdpar C++ and Fortran tutorials
https://github.com/NVIDIA/hpc-container-maker/
"""

import platform
from hpccm.primitives import raw

ubuntu_ver = '22.04'
nvhpc_ver = '24.3'
cuda_ver = '12.3'
gcc_ver = '13'
llvm_ver = '18'
cmake_ver = '3.27.2'
boost_ver = '1.75.0'
arch = platform.machine()

Stage0 += baseimage(image=f'nvcr.io/nvidia/nvhpc:{nvhpc_ver}-devel-cuda{cuda_ver}-ubuntu{ubuntu_ver}')

Stage0 += copy(src='.', dest='/accelerated-computing-hub')
Stage0 += copy(src='brev/update-git-branch.bash', dest='/update-git-branch.bash')

Stage0 += workdir(directory=f'/accelerated-computing-hub/tutorials/stdpar/notebooks')

Stage0 += packages(ospackages=[
  'libtbb-dev',  # Required for GCC C++ parallel algorithms
  'python3', 'python3-pip', 'python-is-python3', 'python3-setuptools', 'python3-dev',
  'make', 'build-essential', 'git', 'git-lfs',
  'curl', 'wget', 'zip', 'bc',
  'nginx', 'openssh-client',
  'libnuma1',  'numactl',
])
Stage0 += boost(version=boost_ver) # Required for AdaptiveCpp

# Install GNU and LLVM toolchains
Stage0 += gnu(version=gcc_ver, extra_repository=True)
Stage0 += llvm(version=llvm_ver, upstream=True, extra_tools=True, toolset=True, _trunk_version='19')

# Patch libstdc++ to use our modified cartesian_product view that doesn't require HMM/ATS and copies
# the underlying range iterators instead of accessing them through host memory. This must be done
# after GCC is installed.
Stage0 += copy(src='tutorials/stdpar/include/ach/cartesian_product.hpp', dest='/usr/include/ach/cartesian_product.hpp')
Stage0 += copy(src='tutorials/stdpar/include/ranges', dest=f'/usr/include/c++/{gcc_ver}/ranges')

# Install CMake
Stage0 += cmake(eula=True, version=cmake_ver)

Stage0 += shell(commands=[
  'set -ex',  # Exit on first error and debug output

  # Configure the HPC SDK toolchain to pick the latest GCC
  f'cd /opt/nvidia/hpc_sdk/Linux_{arch}/{nvhpc_ver}/compilers/bin/',
  'makelocalrc -d . -x .',
  'cd -',

  # Install python packages
  'pip install --upgrade pip',
  f'pip install --root-user-action=ignore -r /accelerated-computing-hub/tutorials/stdpar/brev/requirements.txt',

  # Build and install AdaptiveCpp
  'git clone --depth=1 --shallow-submodules --recurse-submodules -b develop https://github.com/AdaptiveCpp/AdaptiveCpp',
  'cd AdaptiveCpp',
  'git submodule update --recursive',
  f'cmake -Bbuild -H.  -DCMAKE_C_COMPILER="$(which clang-{llvm_ver})" -DCMAKE_CXX_COMPILER="$(which clang++-{llvm_ver})" -DCMAKE_INSTALL_PREFIX=/opt/adaptivecpp -DWITH_CUDA_BACKEND=ON -DWITH_CPU_BACKEND=ON',
  'cmake --build build --target install -j $(nproc)',
  'cd -',
  'rm -rf AdaptiveCpp',

  # Install latest versions of range-v3 and NVIDIA's std::execution implementation
  'git clone --depth=1 https://github.com/ericniebler/range-v3.git',
  'cp -r range-v3/include/* /usr/include/',
  'rm -rf range-v3',
  'git clone --depth=1 https://github.com/nvidia/stdexec.git',
  'cp -r stdexec/include/* /usr/include/',
  'rm -rf stdexec',

  # libc++abi: make sure clang with -stdlib=libc++ can find it
  f'ln -sf /usr/lib/llvm-{llvm_ver}/lib/libc++abi.so.1 /usr/lib/llvm-{llvm_ver}/lib/libc++abi.so',

  # Make mdspan use the paren operator for C++20 compatibility and put it in the std namespace
  f'echo "#define MDSPAN_USE_PAREN_OPERATOR 1"|cat - /opt/nvidia/hpc_sdk/Linux_{arch}/{nvhpc_ver}/compilers/include/experimental/mdspan > /tmp/out && mv /tmp/out /opt/nvidia/hpc_sdk/Linux_{arch}/{nvhpc_ver}/compilers/include/experimental/mdspan',
  f'echo "namespace std {{ using namespace ::std::experimental; }}" >> /opt/nvidia/hpc_sdk/Linux_{arch}/{nvhpc_ver}/compilers/include/experimental/mdspan',

  # Install the NVIDIA HPC SDK mdspan systemwide:
  f'ln -sf /opt/nvidia/hpc_sdk/Linux_{arch}/{nvhpc_ver}/compilers/include/experimental/mdspan /usr/include/mdspan',
  f'ln -sf /opt/nvidia/hpc_sdk/Linux_{arch}/{nvhpc_ver}/compilers/include/experimental/__p0009_bits /usr/include/__p0009_bits',

  # Put the include directory in the systemwide path:
  f'ln -sf /accelerated-computing-hub/tutorials/stdpar/include/ach /usr/include/ach',

  # Don't send GitHub actions CI token when using Git
  'git config --unset-all "http.https://github.com/.extraheader" || { code=$?; [ "$code" = 5 ] || exit "$code"; }',

  # Configure Git to not complain about file ownership
  'git config --global --add safe.directory "/accelerated-computing-hub"',

  # Make sure bash history directory exists
  'mkdir -p ~/.local/state/._bash_history',

  # Configure JupyterLab
  'mkdir -p ~/.jupyter',
  'ln -fs /accelerated-computing-hub/brev/jupyter-server-config.py ~/.jupyter/jupyter_server_config.py',

  # Configure IPython to add the current working directory to the path
  'mkdir -p ~/.ipython/profile_default/startup',
  'ln -fs /accelerated-computing-hub/brev/ipython-startup-add-cwd-to-path.py ~/.ipython/profile_default/startup/00-add-cwd-to-path.py',

  # Silence JupyterLab announcements
  'python -m jupyter labextension disable "@jupyterlab/apputils-extension:announcements"',
])

Stage0 += raw(docker='ARG GIT_BRANCH_NAME')

Stage0 += environment(variables={
  'GIT_BRANCH_NAME': '${GIT_BRANCH_NAME}',

  'ACH_STDPAR_NVHPC_VERSION': nvhpc_ver,
  'ACH_STDPAR_CUDA_VERSION': cuda_ver,
  'ACH_STDPAR_ARCH': arch,

  'ACPP_APPDB_DIR': '/accelerated-computing-hub/',

  'PATH':            '$PATH:/opt/adaptivecpp/bin',
  'LD_LIBRARY_PATH': f'/usr/lib/llvm-{llvm_ver}/lib:$LD_LIBRARY_PATH',
  'LIBRARY_PATH':    f'/usr/lib/llvm-{llvm_ver}/lib:$LIBRARY_PATH',

  # Silence pip warnings about running as root
  'PIP_ROOT_USER_ACTION': 'ignore',

  # Simplify running HPC-X on systems without InfiniBand
  'OMPI_MCA_coll_hcoll_enable': '0',

  # We do not need VFS for the exercises, and using it from a container in a 'generic' way is not trivial:
  'UCX_VFS_ENABLE': 'n',

  # Allow HPC-X to oversubscribe the CPU with more ranks than cores without using mpirun --oversubscribe
  'OMPI_MCA_rmaps_base_oversubscribe' : 'true',

  # Select matplotdir config directory to silence warning
  'MPLCONFIGDIR': '/tmp/matplotlib',

  # Allow OpenMPI to run as root:
  'OMPI_ALLOW_RUN_AS_ROOT': '1',
  'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM': '1',

  # Workaround hwloc binding:
  'OMPI_MCA_hwloc_base_binding_policy': 'none',

  # Workaround nvfortran limit of 64k thread blocks
  'NVCOMPILER_ACC_GANGLIMIT': '67108864', # (1 << 26)
})

Stage0 += raw(docker='ENTRYPOINT ["/accelerated-computing-hub/brev/jupyter-start.bash"]')
