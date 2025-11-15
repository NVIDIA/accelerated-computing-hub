"""
Startup tests for accelerated-python tutorial.
These tests validate that key packages are installed and functional.
"""

import pytest
import numpy as np

def test_cuda_python():
    """Test that cuda-python works by querying device properties using cuda.core."""
    from cuda.core.experimental import system, Device

    # Check CUDA driver version
    assert system.driver_version is not None
    assert len(str(system.driver_version)) > 0

    # Get device count
    assert system.num_devices > 0, "No CUDA devices found"

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
    compute.reduce_into(d_input, d_output, compute.OpKind.PLUS, n, h_init)

    result = float(d_output.get()[0])

    # Verify result (sum of 1000 ones should be 1000)
    assert np.isclose(result, float(n), rtol=1e-5)


def test_cuda_cooperative():
    """Test that cuda.coop works by using block load cooperative algorithm."""
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    import cuda.coop as coop
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
    block_load = coop.block.load(cp.float32, threads_per_block, items_per_thread, 'striped')

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
