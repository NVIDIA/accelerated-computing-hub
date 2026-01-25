import numpy as np
import cupy as cp
from numba import cuda
import nvmath.linalg
from nvmath.bindings import cublas
from common import random_real, mm_perf_GFlops
from common_numba import time_numba, set_max_dynamic_shared_size_bytes
from common_cupy import time_cupy
from emulation_slicing_impl import get_slice_kernel, get_max_reduce_kernel, slice_matrix
from emulation_utils import MatrixHalf


def _calculate_accuracy_metrics(h_result, h_reference):
    """Calculate various accuracy metrics comparing result to reference."""
    result_norm = np.linalg.norm(h_result)
    reference_norm = np.linalg.norm(h_reference)
    
    diff = h_result - h_reference
    abs_diff = np.abs(diff)
    
    # Relative error per element (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = abs_diff / np.abs(h_reference)
        relative_errors = np.where(np.isfinite(relative_errors), relative_errors, 0)
    
    avg_relative_error = np.mean(relative_errors)
    max_relative_error = np.max(relative_errors)
    max_absolute_error = np.max(abs_diff)
    total_relative_error = np.linalg.norm(diff) / reference_norm
    
    return {
        'result_norm': result_norm,
        'reference_norm': reference_norm,
        'avg_relative_error': avg_relative_error,
        'max_relative_error': max_relative_error,
        'max_absolute_error': max_absolute_error,
        'total_relative_error': total_relative_error
    }


def _print_accuracy_metrics(metrics):
    """Print accuracy metrics in standard format."""
    print(f"Vector reference  norm: [{metrics['reference_norm']:.5e}]")
    print(f"Vector result  norm: [{metrics['result_norm']:.5e}]")
    print(f"Vector  relative error: [{metrics['total_relative_error']:.5e}]")
    print(f"Average relative error: [{metrics['avg_relative_error']:.5e}]")
    print(f"Maximum relative error: [{metrics['max_relative_error']:.5e}]")
    print(f"Maximum absolute error: [{metrics['max_absolute_error']:.5e}]")
    print(f"Total relative error = {metrics['total_relative_error']:.10f}")


def _print_performance(time_ms, tflops, label="Custom Kernel"):
    """Print performance metrics in standard format."""
    print(label)
    print(f"Avg time [ms]  = {time_ms:.4f}")
    print(f"Avg TFLOP/s  = {tflops:.4f}\n")


def benchmark_dgemm(
    shapes,
    get_kernel_func,
    kernel_state_func,
    grid_func,
    block_func,
    kernel_args_func=None,
    shared_mem_func=None,
    validation_func=None,
    repeats=10
):
    """
    Generic DGEMM benchmark framework.
    
    Args:
        shapes: List of tuples (m, n, k, alpha, beta) defining problem sizes
        get_kernel_func: Function that returns the kernel. Called as get_kernel_func(state)
        kernel_state_func: Function(m, n, k, alpha, beta) -> state object (can be anything)
        grid_func: Function(m, n, k, state) -> grid dimensions tuple
        block_func: Function(m, n, k, state) -> block dimensions tuple
        kernel_args_func: Optional function(state, alpha, d_A, d_B, beta, d_C) -> tuple of kernel args.
                         Defaults to (alpha, d_A, d_B, beta, d_C).
        shared_mem_func: Optional function(state, kernel_args) -> shared memory size in bytes. Defaults to 0.
        validation_func: Optional function(m, n, k, alpha, beta, block) to validate parameters
        repeats: Number of timing iterations
    """
    for shape in shapes:
        m, n, k, alpha, beta = shape
        
        # Create state that can be shared across kernel/grid/block functions
        state = kernel_state_func(m, n, k, alpha, beta)
        
        # Get block and grid using the state
        block = block_func(m, n, k, state)
        grid = grid_func(m, n, k, state)
        
        print(f"\nComputing GEMM M={m} N={n} K={k}\n")
        
        # Default validation
        if validation_func:
            validation_func(m, n, k, alpha, beta, block)
        else:
            assert m % block[0] == 0, "Invalid M dimension for block size"
            assert n % block[1] == 0, "Invalid N dimension for block size"
            assert k % 16 == 0, "Invalid K dimension"
        
        # Prepare data
        h_A = random_real((m, k), np.float64, order='C')
        h_B = random_real((k, n), np.float64, order='F')
        h_C = random_real((m, n), np.float64, order='F')
        
        d_A_cp = cp.array(h_A)
        d_B_cp = cp.array(h_B)
        d_C_cp = cp.array(h_C)
        
        d_A = cuda.as_cuda_array(d_A_cp)
        d_B = cuda.as_cuda_array(d_B_cp)
        d_C = cuda.as_cuda_array(d_C_cp)
        
        # Get kernel using the state
        dgemm_kernel = get_kernel_func(state)
        
        # Prepare kernel arguments
        if kernel_args_func:
            kernel_args = kernel_args_func(state, alpha, d_A, d_B, beta, d_C)
        else:
            kernel_args = (alpha, d_A, d_B, beta, d_C)
        
        # Calculate shared memory size (after kernel args are created)
        shared_mem_size = shared_mem_func(state, kernel_args) if shared_mem_func else 0
        
        # Set max dynamic shared memory size if needed
        if shared_mem_size > 0:
            set_max_dynamic_shared_size_bytes(dgemm_kernel, shared_mem_size, *kernel_args)
        
        # Run custom kernel once to get result
        dgemm_kernel[grid, block, 0, shared_mem_size](*kernel_args)
        cuda.synchronize()
        
        # Run reference
        d_CRef_cp = cp.array(h_C)
        d_CRef_cp = nvmath.linalg.matmul(d_A_cp, d_B_cp, c=d_CRef_cp, alpha=alpha, beta=beta)
        
        # Calculate accuracy
        h_C = cp.asnumpy(d_C_cp)
        h_CRef = cp.asnumpy(d_CRef_cp)
        metrics = _calculate_accuracy_metrics(h_C, h_CRef)
        
        # Time custom kernel (need to recreate args for timing since they may be modified)
        if kernel_args_func:
            # For timing, we need to pass a function that creates fresh args each time
            def get_timing_args():
                return kernel_args_func(state, alpha, d_A, d_B, beta, d_C)
            # time_numba expects individual args, so we'll need to unpack
            # We'll time with the kernel_args we already have
            time_ms = time_numba(
                dgemm_kernel,
                grid,
                block,
                shared_mem_size,
                repeats,
                *kernel_args)
        else:
            time_ms = time_numba(
                dgemm_kernel,
                grid,
                block,
                shared_mem_size,
                repeats,
                alpha,
                d_A,
                d_B,
                beta,
                d_C)
        
        tflops = mm_perf_GFlops((m, n, k), 1, time_ms) / 1000.0
        _print_performance(time_ms, tflops, "Custom Kernel")
        
        # Time reference (cuBLASLt via nvmath)
        d_CRef2_cp = cp.array(h_C)
        def matmul_ref():
            return nvmath.linalg.matmul(d_A_cp, d_B_cp, c=d_CRef2_cp, alpha=alpha, beta=beta)
        
        ref_time_ms = time_cupy(matmul_ref, repeats)
        ref_tflops = mm_perf_GFlops((m, n, k), 1, ref_time_ms) / 1000.0
        _print_performance(ref_time_ms, ref_tflops, "cuBLASLt (not including heuristic)")
        
        # Print accuracy information
        _print_accuracy_metrics(metrics)
        
        # Calculate performance percentage
        performance_pct = (ref_time_ms / time_ms) * 100.0
        print(f"{performance_pct:.2f}% reference performance ")


def benchmark_emulated_dgemm(
    shapes,
    setup_func,
    gemm_func,
    epilogue_func=None,
    allocate_products=False,
    allocate_diagonals=False,
    num_products_func=None,
    slices=7,
    repeats=10
):
    """
    Benchmark framework for emulated DGEMM.
    
    Handles slicing and allocations, then calls user-provided GEMM and epilogue functions.
    Assumes fixed layout: row_major A, col_major B, col_major C.
    
    Args:
        shapes: List of tuples (m, n, k, alpha, beta) defining problem sizes
        gemm_func: Function(d_A_sliced, d_B_sliced, d_products/d_diag, d_shift_a, d_shift_b) 
                   that performs the GEMM operation
        epilogue_func: Optional function(slices, d_products/d_diag, d_shift_a, d_shift_b, d_C, alpha, beta)
                      for reconstruction. Other dimensions are inferred from tensor shapes.
        allocate_products: If True, allocates products tensor (M×N×num_products) for unfused
        allocate_diagonals: If True, allocates diagonal tensor (M×N×slices) for partial fused
        num_products_func: Optional function(slices) -> int for number of products. 
                          Defaults to slices*(slices+1)//2 if allocate_products=True, else slices.
        slices: Number of slices (default: 7)
        repeats: Number of timing iterations
    """
    for shape in shapes:
        m, n, k, alpha, beta = shape

        context = setup_func(m, n, k)
        
        block_size = 64  # Default block size for slicing kernels
        
        print(f"\nComputing Emulated GEMM M={m} N={n} K={k} (slices={slices})\n")
        
        # Prepare data - fixed layout: row_major A, col_major B, col_major C
        h_A = random_real((m, k), np.float64, order='C')  # row_major
        h_B = random_real((k, n), np.float64, order='F')  # col_major
        h_C = random_real((m, n), np.float64, order='F')  # col_major
        
        d_A_cp = cp.array(h_A)
        d_B_cp = cp.array(h_B)
        d_C_cp = cp.array(h_C)
        
        d_A = cuda.as_cuda_array(d_A_cp)
        d_B = cuda.as_cuda_array(d_B_cp)
        d_C = cuda.as_cuda_array(d_C_cp)
        
        # Allocate sliced tensors with fixed strides
        # A: row_major -> strides = (K, 1, M*K)
        # B: col_major -> strides = (1, K, K*N)
        itemsize = np.dtype(np.int8).itemsize
        strides_a = (k * itemsize, 1 * itemsize, m * k * itemsize)
        
        d_A_sliced_cp = cp.ndarray(
            shape=(m, k, slices),
            dtype=np.int8,
            memptr=cp.cuda.alloc(m * k * slices * itemsize),
            strides=strides_a
        )
        d_A_sliced = cuda.as_cuda_array(d_A_sliced_cp)
        
        strides_b = (1 * itemsize, k * itemsize, k * n * itemsize)
        
        d_B_sliced_cp = cp.ndarray(
            shape=(k, n, slices),
            dtype=np.int8,
            memptr=cp.cuda.alloc(k * n * slices * itemsize),
            strides=strides_b
        )
        d_B_sliced = cuda.as_cuda_array(d_B_sliced_cp)
        
        # Allocate shift tensors
        d_shift_a = cuda.device_array(m, dtype=np.int32)
        d_shift_b = cuda.device_array(n, dtype=np.int32)
        
        # Allocate products or diagonal tensor based on flags
        itemsize_int32 = np.dtype(np.int32).itemsize
        d_products = None
        d_diagonals = None
        
        if allocate_products:
            # Allocate products tensor for unfused (M×N×num_products)
            num_products = num_products_func(slices) if num_products_func else slices * (slices + 1) // 2
            # C is col_major -> strides = (1, M, M*N)
            strides_products = (1 * itemsize_int32, m * itemsize_int32, m * n * itemsize_int32)
            
            d_products_cp = cp.ndarray(
                shape=(m, n, num_products),
                dtype=np.int32,
                memptr=cp.cuda.alloc(m * n * num_products * itemsize_int32),
                strides=strides_products
            )
            d_gemm_out_cp = d_products_cp
            d_gemm_out_numba = cuda.as_cuda_array(d_products_cp)
        elif allocate_diagonals:
            # Allocate diagonal tensor for partial fusion (M×N×slices)
            num_diag = num_products_func(slices) if num_products_func else slices
            # C is col_major -> strides = (1, M, M*N)
            strides_diag = (1 * itemsize_int32, m * itemsize_int32, m * n * itemsize_int32)
            
            d_diag_cp = cp.ndarray(
                shape=(m, n, num_diag),
                dtype=np.int32,
                memptr=cp.cuda.alloc(m * n * num_diag * itemsize_int32),
                strides=strides_diag
            )
            d_gemm_out_cp = d_diag_cp
            d_gemm_out_numba = cuda.as_cuda_array(d_diag_cp)
        else:
            d_gemm_out_cp = d_C_cp
            d_gemm_out_numba = d_C
        
        # Get slicing kernels
        max_reduce_a = get_max_reduce_kernel(slice_matrix.a, BlockSize=block_size)
        max_reduce_b = get_max_reduce_kernel(slice_matrix.b, BlockSize=block_size)
        slice_kernel_a = get_slice_kernel(slice_matrix.a, Slices=slices, BlockSize=block_size)
        slice_kernel_b = get_slice_kernel(slice_matrix.b, Slices=slices, BlockSize=block_size)
        
        # Perform slicing
        blocks_a = m
        blocks_b = n
        max_reduce_a[blocks_a, block_size](d_A, d_shift_a)
        max_reduce_b[blocks_b, block_size](d_B, d_shift_b)
        
        num_elements_a = m * k
        num_elements_b = k * n
        blocks_slice_a = (num_elements_a + block_size - 1) // block_size
        blocks_slice_b = (num_elements_b + block_size - 1) // block_size
        
        slice_kernel_a[blocks_slice_a, block_size](d_A, d_shift_a, d_A_sliced, k)
        slice_kernel_b[blocks_slice_b, block_size](d_B, d_shift_b, d_B_sliced, n)
        
        # Run GEMM function (pass CuPy arrays)
        gemm_func(d_A_sliced_cp, d_B_sliced_cp, d_gemm_out_cp, d_shift_a, d_shift_b, alpha, beta, context)
        
        # Run epilogue if provided (pass Numba arrays for kernel compatibility)
        if epilogue_func:
            epilogue_func(slices, d_gemm_out_numba, d_shift_a, d_shift_b, d_C, alpha, beta, context)
        
        cuda.synchronize()
        
        # Run reference
        d_CRef_cp = cp.array(h_C)
        d_CRef_cp = nvmath.linalg.matmul(d_A_cp, d_B_cp, c=d_CRef_cp, alpha=alpha, beta=beta)
        
        # Calculate accuracy
        h_C = cp.asnumpy(d_C_cp)
        h_CRef = cp.asnumpy(d_CRef_cp)
        metrics = _calculate_accuracy_metrics(h_C, h_CRef)
        
        # Time slicing separately
        # Start with a warmup
        max_reduce_a[blocks_a, block_size](d_A, d_shift_a)
        max_reduce_b[blocks_b, block_size](d_B, d_shift_b)
        slice_kernel_a[blocks_slice_a, block_size](d_A, d_shift_a, d_A_sliced, k)
        slice_kernel_b[blocks_slice_b, block_size](d_B, d_shift_b, d_B_sliced, n)
        
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(repeats):
            max_reduce_a[blocks_a, block_size](d_A, d_shift_a)
            max_reduce_b[blocks_b, block_size](d_B, d_shift_b)
            slice_kernel_a[blocks_slice_a, block_size](d_A, d_shift_a, d_A_sliced, k)
            slice_kernel_b[blocks_slice_b, block_size](d_B, d_shift_b, d_B_sliced, n)
        end.record()
        end.synchronize()
        slicing_time_ms = cp.cuda.get_elapsed_time(start, end) / repeats

        # Warmup GEMM
        gemm_func(d_A_sliced_cp, d_B_sliced_cp, d_gemm_out_cp, d_shift_a, d_shift_b, alpha, beta, context)
        
        # Time GEMM separately
        start.record()
        for _ in range(repeats):
            gemm_func(d_A_sliced_cp, d_B_sliced_cp, d_gemm_out_cp, d_shift_a, d_shift_b, alpha, beta, context, warmup=False)
        end.record()
        end.synchronize()
        gemm_time_ms = cp.cuda.get_elapsed_time(start, end) / repeats
        
        # Time epilogue separately (if present)
        epilogue_time_ms = 0.0
        if epilogue_func:
            # Warmup epilogue
            epilogue_func(slices, d_gemm_out_numba, d_shift_a, d_shift_b, d_C, alpha, beta, context)
            start.record()
            for _ in range(repeats):
                epilogue_func(slices, d_gemm_out_numba, d_shift_a, d_shift_b, d_C, alpha, beta, context)
            end.record()
            end.synchronize()
            epilogue_time_ms = cp.cuda.get_elapsed_time(start, end) / repeats
        
        # Calculate E2E time
        e2e_time_ms = slicing_time_ms + gemm_time_ms + epilogue_time_ms
        
        # Print timing breakdown
        print("Emulated GEMM Timing Breakdown")
        print(f"Slicing time [ms]   = {slicing_time_ms:.4f}")
        print(f"GEMM time [ms]      = {gemm_time_ms:.4f}")
        if epilogue_func:
            print(f"Epilogue time [ms]  = {epilogue_time_ms:.4f}")
        print(f"E2E time [ms]       = {e2e_time_ms:.4f}")
        
        e2e_tflops = mm_perf_GFlops((m, n, k), 1, e2e_time_ms) / 1000.0
        print(f"E2E TFLOP/s         = {e2e_tflops:.4f}\n")
        
        # Time reference (cuBLASLt via nvmath)
        d_CRef2_cp = cp.array(h_C)
        def matmul_ref():
            return nvmath.linalg.matmul(d_A_cp, d_B_cp, c=d_CRef2_cp, alpha=alpha, beta=beta)
        
        ref_time_ms = time_cupy(matmul_ref, repeats)
        ref_tflops = mm_perf_GFlops((m, n, k), 1, ref_time_ms) / 1000.0
        _print_performance(ref_time_ms, ref_tflops, "cuBLASLt (not including heuristic)")
        
        # Print accuracy information
        _print_accuracy_metrics(metrics)
        
        # Calculate performance percentage
        performance_pct = (ref_time_ms / e2e_time_ms) * 100.0
        print(f"{performance_pct:.2f}% reference performance ")


# Convenience wrappers for specific exercises
def benchmark_dgemm_2_1(shapes, get_dgemm_kernel, choose_kernel_params, repeats=10):
    """Exercise 2.1: Variable block size, kernel factory takes block dimensions."""
    return benchmark_dgemm(
        shapes=shapes,
        get_kernel_func=lambda state: get_dgemm_kernel(*state),
        kernel_state_func=lambda m, n, k, alpha, beta: choose_kernel_params(m, n, k, alpha, beta),
        grid_func=lambda m, n, k, state: (m // state[0], n // state[1]),
        block_func=lambda m, n, k, state: state,
        repeats=repeats
    )


def benchmark_dgemm_2_2(shapes, get_dgemm_kernel, repeats=10):
    """Exercise 2.2: Fixed 16x16 block size, kernel factory takes no arguments."""
    return benchmark_dgemm(
        shapes=shapes,
        get_kernel_func=lambda state: get_dgemm_kernel(),
        kernel_state_func=lambda m, n, k, alpha, beta: (16, 16),
        grid_func=lambda m, n, k, state: (m // state[0], n // state[1]),
        block_func=lambda m, n, k, state: state,
        repeats=repeats
    )


def benchmark_dgemm_2_3(shapes, get_dgemm_kernel, choose_kernel_params, shared_mem_func, kernel_args_func, repeats=10):
    """Exercise 2.4: Using cublasDx with pipeline API."""

    def validation_func(m, n, k, alpha, beta, block):
        pass
    
    return benchmark_dgemm(
        shapes=shapes,
        get_kernel_func=get_dgemm_kernel,
        kernel_state_func=lambda m, n, k, alpha, beta: choose_kernel_params(m, n, k, alpha, beta),
        grid_func=lambda m, n, k, BLAS: (m // BLAS.c_dim[0], n // BLAS.c_dim[1]),
        block_func=lambda m, n, k, BLAS: (BLAS.block_size, 1),
        kernel_args_func=kernel_args_func,
        shared_mem_func=shared_mem_func,
        validation_func=validation_func,
        repeats=repeats
    )


def benchmark_dgemm_2_4(shapes, get_dgemm_kernel, choose_kernel_params, shared_mem_func, repeats=10):
    """Exercise 2.3: Using cublasDx."""

    def validation_func(m, n, k, alpha, beta, block):
        pass
    
    return benchmark_dgemm(
        shapes=shapes,
        get_kernel_func=get_dgemm_kernel,
        kernel_state_func=lambda m, n, k, alpha, beta: choose_kernel_params(m, n, k, alpha, beta),
        grid_func=lambda m, n, k, BLAS: (m // BLAS.c_dim[0], n // BLAS.c_dim[1]),
        block_func=lambda m, n, k, BLAS: (BLAS.block_size, 1),
        shared_mem_func=lambda state, kernel_args: shared_mem_func(state),
        validation_func=validation_func,
        repeats=repeats
    )

def benchmark_unfused_emulated_dgemm(shapes, setup_func, igemm_func, epilogue_func, slices=7, repeats=10):
    """
    Benchmark unfused emulated DGEMM.
    
    Unfused implementation:
    - Computes all products (slices*(slices+1)//2) separately
    - Stores products in M×N×num_products tensor
    - Epilogue reconstructs the final result from products
    
    Args:
        shapes: List of tuples (m, n, k, alpha, beta)
        igemm_func: Function(d_A_sliced, d_B_sliced, d_products)
                   that computes all integer GEMM products
        epilogue_func: Function(slices, d_products, d_shift_a, d_shift_b, d_C, alpha, beta)
                      that reconstructs the final result
        slices: Number of slices (default: 7)
        repeats: Number of timing iterations
    """
    # Wrap igemm_func to ignore shift arguments
    gemm_wrapper = lambda a, b, prod, shift_a, shift_b, alpha, beta, context, warmup=True: igemm_func(a, b, prod, context, warmup)
    
    return benchmark_emulated_dgemm(
        shapes=shapes,
        setup_func=setup_func,
        gemm_func=gemm_wrapper,
        epilogue_func=epilogue_func,
        allocate_products=True,
        allocate_diagonals=False,
        num_products_func=lambda s: s * (s + 1) // 2,
        slices=slices,
        repeats=repeats
    )


def benchmark_partially_fused_emulated_dgemm(shapes, setup_func, igemm_func, epilogue_func, slices=7, repeats=10):
    """
    Benchmark partially fused emulated DGEMM.
    
    Partially fused implementation:
    - Computes and partially sums products into diagonal tensor
    - Stores diagonals in M×N×slices tensor (one per slice combination diagonal)
    - Epilogue reconstructs the final result from diagonals
    
    Args:
        shapes: List of tuples (m, n, k, alpha, beta)
        igemm_func: Function(d_A_sliced, d_B_sliced, d_diagonals)
                   that computes diagonal sums
        epilogue_func: Function(slices, d_diagonals, d_shift_a, d_shift_b, d_C, alpha, beta)
                      that reconstructs the final result
        slices: Number of slices (default: 7)
        repeats: Number of timing iterations
    """
    # Wrap igemm_func to ignore shift arguments
    gemm_wrapper = lambda a, b, diag, shift_a, shift_b, alpha, beta, context, warmup=True: igemm_func(a, b, diag, context, warmup)
    
    return benchmark_emulated_dgemm(
        shapes=shapes,
        setup_func=setup_func,
        gemm_func=gemm_wrapper,
        epilogue_func=epilogue_func,
        allocate_products=False,
        allocate_diagonals=True,
        num_products_func=lambda s: s,
        slices=slices,
        repeats=repeats
    )


def benchmark_fused_emulated_dgemm(shapes, setup_func, fused_kernel_func, slices=7, repeats=10):
    """
    Benchmark fully fused emulated DGEMM.
    
    Fully fused implementation:
    - Everything computed in a single kernel
    - No intermediate storage (products/diagonals)
    - Directly writes final result to d_C
    
    Args:
        shapes: List of tuples (m, n, k, alpha, beta)
        fused_kernel_func: Function(d_A_sliced, d_B_sliced, None, d_shift_a, d_shift_b)
                          that performs the entire operation
        slices: Number of slices (default: 7)
        repeats: Number of timing iterations
    """
    return benchmark_emulated_dgemm(
        shapes=shapes,
        setup_func=setup_func,
        gemm_func=fused_kernel_func,
        epilogue_func=None,
        allocate_products=False,
        allocate_diagonals=False,
        slices=slices,
        repeats=repeats
    )


def benchmark_fused_emulated_dsyrk(shapes, setup_func, fused_kernel_func, slices=7, repeats=10):
    """
    Benchmark fully fused emulated DSYRK (Symmetric Rank-K Update).
    
    SYRK computes: C = alpha * A @ A^T + beta * C where C is symmetric.
    Only supports the fused variant (no intermediate products/diagonals).
    Assumes fixed layout: row_major A, col_major C.
    
    Args:
        shapes: List of tuples (n, k, alpha, beta, uplo) defining problem sizes
                n: dimension of square output C (N×N)
                k: inner dimension of A (N×K)
                uplo: MatrixHalf.lower or MatrixHalf.upper (or 'L'/'U' strings for convenience)
        setup_func: Function(n, k, matrix_half) -> context dict
        fused_kernel_func: Function(d_A_sliced, d_C, d_shift_a, alpha, beta, context)
                          that performs the entire SYRK operation
        slices: Number of slices (default: 7)
        repeats: Number of timing iterations
    """
    # Create cuBLAS handle once for all problem sizes
    handle = cublas.create()
    
    try:
        for shape in shapes:
            n, k, alpha, beta, uplo = shape
            
            # Convert uplo string to MatrixHalf enum
            if uplo == 'L' or uplo == MatrixHalf.lower:
                matrix_half = MatrixHalf.lower
                uplo_str = 'L'
            elif uplo == 'U' or uplo == MatrixHalf.upper:
                matrix_half = MatrixHalf.upper
                uplo_str = 'U'
            else:
                raise ValueError(f"Invalid uplo: {uplo}. Must be 'L', 'U', MatrixHalf.lower, or MatrixHalf.upper")

            context = setup_func(n, k, matrix_half)
            
            block_size = 64  # Default block size for slicing kernels
            
            print(f"\nComputing Emulated SYRK N={n} K={k} (slices={slices}, uplo={uplo_str})\n")
            
            # Prepare data - fixed layout: row_major A, col_major C
            h_A = random_real((n, k), np.float64, order='C')  # row_major
            h_C = random_real((n, n), np.float64, order='F')  # col_major, square
            h_C_original = h_C.copy()  # Save original C for checking unchanged elements
            
            d_A_cp = cp.array(h_A)
            d_C_cp = cp.array(h_C)
            
            d_A = cuda.as_cuda_array(d_A_cp)
            d_C = cuda.as_cuda_array(d_C_cp)
            
            # Allocate sliced tensors with fixed strides
            # A: row_major -> strides = (K, 1, N*K)
            itemsize = np.dtype(np.int8).itemsize
            strides_a = (k * itemsize, 1 * itemsize, n * k * itemsize)
            
            d_A_sliced_cp = cp.ndarray(
                shape=(n, k, slices),
                dtype=np.int8,
                memptr=cp.cuda.alloc(n * k * slices * itemsize),
                strides=strides_a
            )
            d_A_sliced = cuda.as_cuda_array(d_A_sliced_cp)
            
            # Allocate shift tensors (only need one for A)
            d_shift_a = cuda.device_array(n, dtype=np.int32)
            
            # Get slicing kernels (only need for A)
            max_reduce_a = get_max_reduce_kernel(slice_matrix.a, BlockSize=block_size)
            slice_kernel_a = get_slice_kernel(slice_matrix.a, Slices=slices, BlockSize=block_size)
            
            # Perform slicing
            blocks_a = n
            max_reduce_a[blocks_a, block_size](d_A, d_shift_a)
            
            num_elements_a = n * k
            blocks_slice_a = (num_elements_a + block_size - 1) // block_size
            
            slice_kernel_a[blocks_slice_a, block_size](d_A, d_shift_a, d_A_sliced, k)
            
            # Run fused SYRK kernel
            fused_kernel_func(d_A_sliced_cp, d_C_cp, d_shift_a, alpha, beta, context)
        
            cuda.synchronize()
            
            # Run reference using cuBLAS SYRK
            d_CRef_cp = cp.array(h_C)
            
            # cuBLAS parameters for row-major A
            # uplo: 0 = CUBLAS_FILL_MODE_LOWER, 1 = CUBLAS_FILL_MODE_UPPER  
            # trans: 1 = CUBLAS_OP_T (transpose)
            # For row-major A (N×K): tell cuBLAS to transpose, so it computes A @ A^T
            cublas_uplo = 0 if matrix_half == MatrixHalf.lower else 1
            cublas_trans = 1  # transpose (because A is row-major)
            lda = k  # leading dimension for row-major N×K is k (stride between rows)
            ldc = n  # leading dimension for col-major C
            
            # Create scalar pointers for alpha and beta (on host for cuBLAS)
            alpha_host = np.array([alpha], dtype=np.float64)
            beta_host = np.array([beta], dtype=np.float64)
            alpha_ptr = alpha_host.ctypes.data
            beta_ptr = beta_host.ctypes.data
            
            cublas.dsyrk(
                handle,
                cublas_uplo,
                cublas_trans,
                n,
                k,
                alpha_ptr,
                d_A_cp.data.ptr,
                lda,
                beta_ptr,
                d_CRef_cp.data.ptr,
                ldc
            )
            cp.cuda.Stream.null.synchronize()
            
            # Calculate accuracy (only for the relevant triangular part)
            h_C = cp.asnumpy(d_C_cp)
            h_CRef = cp.asnumpy(d_CRef_cp)
            
            # Create mask for the triangular part that should be updated
            if matrix_half == MatrixHalf.lower:
                # Lower triangle (including diagonal): i >= j
                mask = np.tri(n, n, k=0, dtype=bool)
            else:
                # Upper triangle (including diagonal): i <= j
                mask = np.tri(n, n, k=0, dtype=bool).T
            
            # Check 1: Accuracy metrics only for the triangular part
            h_C_tri = h_C[mask]
            h_CRef_tri = h_CRef[mask]
            metrics = _calculate_accuracy_metrics(h_C_tri, h_CRef_tri)
            
            # Check 2: Verify that elements outside the triangular part haven't been modified
            outside_mask = ~mask
            h_C_outside = h_C[outside_mask]
            h_C_orig_outside = h_C_original[outside_mask]
            
            max_outside_change = np.max(np.abs(h_C_outside - h_C_orig_outside))
            num_changed_outside = np.sum(np.abs(h_C_outside - h_C_orig_outside) > 1e-15)
            
            print(f"\nTriangular Check:")
            print(f"Elements that should be unchanged: {np.sum(outside_mask)}")
            print(f"Elements modified outside triangle: {num_changed_outside}")
            print(f"Max change outside triangle: {max_outside_change:.5e}\n")
            
            # Time slicing separately
            # Start with a warmup
            max_reduce_a[blocks_a, block_size](d_A, d_shift_a)
            slice_kernel_a[blocks_slice_a, block_size](d_A, d_shift_a, d_A_sliced, k)
            
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            for _ in range(repeats):
                max_reduce_a[blocks_a, block_size](d_A, d_shift_a)
                slice_kernel_a[blocks_slice_a, block_size](d_A, d_shift_a, d_A_sliced, k)
            end.record()
            end.synchronize()
            slicing_time_ms = cp.cuda.get_elapsed_time(start, end) / repeats

            # Warmup fused kernel
            fused_kernel_func(d_A_sliced_cp, d_C_cp, d_shift_a, alpha, beta, context)
            
            # Time fused kernel separately
            start.record()
            for _ in range(repeats):
                fused_kernel_func(d_A_sliced_cp, d_C_cp, d_shift_a, alpha, beta, context, warmup=False)
            end.record()
            end.synchronize()
            kernel_time_ms = cp.cuda.get_elapsed_time(start, end) / repeats
            
            # Calculate E2E time
            e2e_time_ms = slicing_time_ms + kernel_time_ms
            
            # Print timing breakdown
            print("Emulated SYRK Timing Breakdown")
            print(f"Slicing time [ms]   = {slicing_time_ms:.4f}")
            print(f"Kernel time [ms]    = {kernel_time_ms:.4f}")
            print(f"E2E time [ms]       = {e2e_time_ms:.4f}")
            
            # SYRK FLOPs: only computing triangular part (n*(n+1)/2 elements)
            # Each element requires k multiply-adds, and each multiply-add is 2 FLOPs
            syrk_flops = n * (n + 1) * k  # (n*(n+1)/2) elements * k * 2 ops
            e2e_tflops = (syrk_flops / 1e12) / (e2e_time_ms / 1000.0)
            print(f"E2E TFLOP/s         = {e2e_tflops:.4f}\n")
            
            # Time reference (using cuBLAS SYRK)
            # Time reference separately using cuBLAS SYRK
            d_CRef2_cp = cp.array(h_C)
            alpha_host_ref = np.array([alpha], dtype=np.float64)
            beta_host_ref = np.array([beta], dtype=np.float64)
            alpha_ptr_ref = alpha_host_ref.ctypes.data
            beta_ptr_ref = beta_host_ref.ctypes.data
            
            # Warmup
            cublas.dsyrk(handle, cublas_uplo, cublas_trans, n, k, 
                         alpha_ptr_ref, d_A_cp.data.ptr, lda,
                         beta_ptr_ref, d_CRef2_cp.data.ptr, ldc)
            cp.cuda.Stream.null.synchronize()
            
            # Reset for timing
            d_CRef2_cp = cp.array(h_C)
            
            # Time cuBLAS SYRK
            start.record()
            for _ in range(repeats):
                cublas.dsyrk(handle, cublas_uplo, cublas_trans, n, k,
                             alpha_ptr_ref, d_A_cp.data.ptr, lda,
                             beta_ptr_ref, d_CRef2_cp.data.ptr, ldc)
            end.record()
            end.synchronize()
            ref_time_ms = cp.cuda.get_elapsed_time(start, end) / repeats
            
            ref_tflops = (syrk_flops / 1e12) / (ref_time_ms / 1000.0)
            _print_performance(ref_time_ms, ref_tflops, "cuBLAS SYRK reference")
            
            # Print accuracy information
            _print_accuracy_metrics(metrics)
            
            # Calculate performance percentage
            performance_pct = (ref_time_ms / e2e_time_ms) * 100.0
            print(f"{performance_pct:.2f}% reference performance ")
    finally:
        # Destroy handle after all problem sizes are done
        cublas.destroy(handle)
