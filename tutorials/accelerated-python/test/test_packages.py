"""
Startup tests for accelerated-python tutorial.
These tests validate that key packages are installed and functional.
"""

import importlib.util
from importlib.metadata import version as distribution_version
import os
import subprocess
import sys

import numpy as np


def test_dependency_versions():
    """Critical Python CUDA packages match the resolved CUDA 13.2 stack."""
    expected = {
        "torch": "2.14.0+cu132",
        "cuda-toolkit": "13.2.1",
        "cuda-core": "1.2.0",
        "cuda-cccl": "1.1.1",
        "cupy-cuda13x": "14.2.0",
        "nvmath-python": "1.0.0",
        "nvidia-nvjitlink": "13.4.92",
        "numba-cuda": "0.30.4",
        "scipy": "1.17.1",
    }
    for distribution, expected_version in expected.items():
        assert distribution_version(distribution) == expected_version

    nvcc = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "release 13.2" in nvcc.stdout


def test_system_environment():
    """The shared system environment uses the current stack and OpenMPI."""
    import jax
    import llvmlite
    import numba

    assert sys.prefix == sys.base_prefix == "/usr"
    assert numba.__version__ == "0.63.1"
    assert llvmlite.__version__ == "0.46.0"
    assert jax.__version__ == "0.11.1"
    assert importlib.util.find_spec("numba.openmp") is not None

    program = (
        "from mpi4py import MPI\n"
        "assert MPI.get_vendor()[0] == 'Open MPI'\n"
        "assert MPI.COMM_WORLD.Get_size() == 2\n"
    )
    default_mpi_env = os.environ.copy()
    default_mpi_env.pop("MPI4PY_MPIABI", None)
    result = subprocess.run(
        [
            "/usr/bin/mpirun.openmpi",
            "--mca",
            "plm",
            "isolated",
            "--oversubscribe",
            "-n",
            "2",
            sys.executable,
            "-c",
            program,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=default_mpi_env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_python():
    """Test that cuda-python works by querying device properties using cuda.core."""
    from cuda.core import system, Device

    # Check CUDA driver version
    driver_version = system.get_kernel_mode_driver_version()
    assert driver_version is not None
    assert len(str(driver_version)) > 0

    # Get device count
    assert system.get_num_devices() > 0, "No CUDA devices found"

    # Get device information
    device = Device(0)
    device.set_current()

    # Verify device properties
    assert len(device.name) > 0
    assert device.device_id == 0
    assert device.uuid is not None
    assert device.pci_bus_id is not None


def test_numba_cuda():
    """Test that numba CUDA works by JIT compiling and running a simple kernel."""
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    from numba import cuda
    import numpy as np

    # Suppress grid size performance warnings for this test
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    assert cuda.is_available(), "CUDA not available for numba"

    # Define a simple kernel that adds two arrays
    @cuda.jit
    def add_kernel(a, b, c):
        i = cuda.grid(1)
        if i < a.size:
            c[i] = a[i] + b[i]

    # Create test data
    n = 100
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)

    # Copy to device and run kernel
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    threads_per_block = 32
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result back
    result = d_c.copy_to_host()

    # Verify result
    assert np.allclose(result, 2.0)


def test_cuda_compute():
    """Test that cuda.compute works by running a parallel reduction algorithm."""
    import cuda.compute as compute
    import cupy as cp

    # Test parallel reduce operation using cuda.compute
    # Create test data on GPU
    n = 1000
    d_input = cp.ones(n, dtype=cp.float32)
    d_output = cp.zeros(1, dtype=cp.float32)
    h_init = np.array([0], dtype=np.float32)

    # Use cuda.compute to reduce (sum) the array
    # reduce_into computes a reduction and stores result in d_output
    compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        op=compute.OpKind.PLUS,
        num_items=n,
        h_init=h_init,
    )

    result = float(d_output.get()[0])

    # Verify result (sum of 1000 ones should be 1000)
    assert np.isclose(result, float(n), rtol=1e-5)


def test_cuda_compute_algorithms():
    """The keyword-only cuda.compute 1.1 APIs execute representative work."""
    import cuda.compute as compute
    import cupy as cp

    d_input = cp.asarray([3, 1, 4, 1], dtype=cp.int32)
    h_init = np.asarray([0], dtype=np.int32)

    d_scan = cp.empty_like(d_input)
    compute.inclusive_scan(
        d_in=d_input,
        d_out=d_scan,
        op=compute.OpKind.PLUS,
        init_value=h_init,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_scan.get(), [3, 4, 8, 9])

    d_merge = cp.empty_like(d_input)
    compute.merge_sort(
        d_in_keys=d_input,
        d_out_keys=d_merge,
        op=compute.OpKind.LESS,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_merge.get(), [1, 1, 3, 4])

    d_radix = cp.empty_like(d_input)
    compute.radix_sort(
        d_in_keys=d_input,
        d_out_keys=d_radix,
        order=compute.SortOrder.DESCENDING,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_radix.get(), [4, 3, 1, 1])

    def double(value):
        return value * 2

    d_unary = cp.empty_like(d_input)
    compute.unary_transform(
        d_in=d_input,
        d_out=d_unary,
        op=double,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_unary.get(), [6, 2, 8, 2])

    d_binary = cp.empty_like(d_input)
    compute.binary_transform(
        d_in1=d_input,
        d_in2=d_input,
        d_out=d_binary,
        op=compute.OpKind.PLUS,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_binary.get(), [6, 2, 8, 2])


def test_cuda_cooperative():
    """Test cuda.coop._experimental with a block-load algorithm."""
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    from cuda.coop._experimental import block
    from numba import cuda
    import cupy as cp

    # Suppress grid size performance warnings for this test
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    assert cuda.is_available(), "CUDA not available for cuda.coop test"

    # Test cuda.coop by creating a block load algorithm and using it in a kernel
    # This tests that cuda.coop can compile cooperative algorithms
    threads_per_block = 32
    items_per_thread = 4
    items_per_block = threads_per_block * items_per_thread

    # Create a cooperative block load algorithm
    block_load = block.make_load(
        cp.float32, threads_per_block, items_per_thread, "striped"
    )

    # Define a kernel that uses the cooperative block load
    @cuda.jit(link=block_load.files)
    def load_and_sum_kernel(input_data, output):
        # Local storage for items loaded by this thread
        items = cuda.local.array(items_per_thread, dtype=input_data.dtype)

        # Use cooperative block load
        base = cuda.blockIdx.x * items_per_block
        block_load(input_data[base : base + items_per_block], items)

        # Sum the loaded items for this thread
        thread_sum = 0.0
        for i in range(items_per_thread):
            thread_sum += items[i]

        # Write to output (one value per thread)
        tid = cuda.grid(1)
        if tid < len(output):
            output[tid] = thread_sum

    # Create test data
    n = items_per_block
    input_data = cp.ones(n, dtype=cp.float32)
    output = cp.zeros(threads_per_block, dtype=cp.float32)

    # Launch kernel
    load_and_sum_kernel[1, threads_per_block](input_data, output)

    # Verify - each thread should have summed items_per_thread ones
    result = output.get()
    assert np.allclose(result, items_per_thread)


def test_cupy():
    """Test that CuPy works by performing array operations on GPU."""
    import cupy as cp

    # Create CuPy arrays on GPU
    x = cp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = cp.array([2.0, 2.0, 2.0, 2.0, 2.0])

    # Perform element-wise operations
    z = x * y
    assert cp.allclose(z, cp.array([2.0, 4.0, 6.0, 8.0, 10.0]))

    # Test reduction
    sum_result = cp.sum(x)
    assert cp.isclose(sum_result, 15.0)

    # Test matrix operations
    a = cp.array([[1.0, 2.0], [3.0, 4.0]])
    b = cp.array([[5.0, 6.0], [7.0, 8.0]])
    c = cp.matmul(a, b)
    expected = cp.array([[19.0, 22.0], [43.0, 50.0]])
    assert cp.allclose(c, expected)


def test_nvmath():
    """nvmath uses the system CUDA libraries for a GPU matrix product."""
    import cupy as cp
    import nvmath

    identity = cp.eye(2, dtype=cp.float32)
    result = nvmath.linalg.advanced.matmul(identity, identity)
    assert cp.allclose(result, identity)


def test_pytorch():
    """Test that PyTorch works by performing tensor operations and checking CUDA."""
    import torch

    # Create tensors and perform operations
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # Matrix multiplication
    z = torch.matmul(x, y.T)

    # Verify result shape and computation
    assert z.shape == (2, 2)
    expected = torch.tensor([[17.0, 23.0], [39.0, 53.0]])
    assert torch.allclose(z, expected)

    # Test autograd
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = (a ** 2).sum()
    b.backward()

    # Gradient should be 2*a = [4.0, 6.0]
    assert torch.allclose(a.grad, torch.tensor([4.0, 6.0]))

    # Verify CUDA is available - this is required
    assert torch.cuda.is_available(), "PyTorch CUDA support is not available"

    # Test device transfer
    device = torch.device('cuda:0')
    x_cuda = x.to(device)
    assert x_cuda.device.type == 'cuda'

    # Test a simple operation on GPU
    y_cuda = y.to(device)
    z_cuda = torch.matmul(x_cuda, y_cuda.T)
    assert z_cuda.device.type == 'cuda'
    assert torch.allclose(z_cuda.cpu(), expected)

def test_nsightful():
    from nsightful.notebook import is_interactive_notebook

    assert not is_interactive_notebook(), "nsightful interactive notebook check failed"
