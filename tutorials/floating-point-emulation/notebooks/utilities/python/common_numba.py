# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from cuda.bindings import driver as cudadrv
from numba import cuda
from numba.cuda.typing import typeof as cuda_typeof
import math
from nvmath.device import float16x2

import _nvmath_fixup

def time_numba(kernel, grid_dim, block_dim, shared_memory_size, ncycles, *args, get_results=None):
    ## Numba
    stream = cuda.stream()
    start, stop = cuda.event(), cuda.event()
    cuda.synchronize()

    # jit + set max dynamic smem size
    set_max_dynamic_shared_size_bytes(kernel, shared_memory_size, *args)

    # warmup
    kernel[grid_dim, block_dim, stream, shared_memory_size](*args)
    stream.synchronize()

    if get_results is not None:
        get_results()

    # time
    start.record(stream)
    for _ in range(ncycles):
        kernel[grid_dim, block_dim, stream, shared_memory_size](*args)
    stop.record(stream)
    stream.synchronize()

    time_ms = cuda.event_elapsed_time(start, stop) / ncycles
    return time_ms


def get_active_blocks_per_multiprocessor(kernel, block_dim, dynamic_smem_size, *args):
    argsty = tuple([cuda_typeof.typeof(a) for a in args])
    compiled = kernel.compile(argsty)
    ctx = cuda.current_context()
    cufunc = compiled.library.get_cufunc()
    active_per_sm = ctx.get_active_blocks_per_multiprocessor(cufunc, math.prod(block_dim), dynamic_smem_size)

    return active_per_sm


def set_max_dynamic_shared_size_bytes(kernel, max_dynamic_smem_size, *args):
    argsty = tuple([cuda_typeof.typeof(a) for a in args])
    compiled = kernel.compile(argsty)
    cufunc = compiled.library.get_cufunc()
    # Starting in numba-cuda 0.15, there are two bindings backends, we need to handle
    # both. See docs about NUMBA_CUDA_USE_NVIDIA_BINDING environment variable.
    if isinstance(cufunc.handle, ctypes.c_void_p):
        handle = cufunc.handle.value
    elif isinstance(cufunc.handle, cudadrv.CUkernel):
        resp, func = cudadrv.cuKernelGetFunction(cufunc.handle)
        if resp != cudadrv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuKernelGetFunction failed with error code {resp}")
        handle = func
    elif isinstance(cufunc.handle, cudadrv.CUfunction):
        handle = cufunc.handle
    else:
        raise RuntimeError(f"Unsupported cufunc.handle type: {type(cufunc.handle)}")

    resp = cudadrv.cuFuncSetAttribute(
        handle,
        cudadrv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        max_dynamic_smem_size,
    )
    if resp[0] != cudadrv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuFuncSetAttribute failed with error code {resp}")


# matrix is always in C-order (cupy/numpy) but smem should always be in F-order (expected by
# cuBLASDx)
@cuda.jit(device=True, forceinline=True)
def load_to_shared_batched(matrix, smem, batch, dim, ld, row_major=False):
    start = cuda.threadIdx.x
    step = cuda.blockDim.x
    stop = dim[0] * dim[1]
    for index in range(start, stop, step):
        col = index % dim[1]
        row = index // dim[1]
        if row_major:
            smem[batch * dim[1] * ld + row * ld + col] = matrix[batch, row, col]
        else:
            smem[batch * dim[1] * ld + col * ld + row] = matrix[batch, row, col]


@cuda.jit(device=True, forceinline=True)
def load_to_shared(matrix, smem, dim, ld, row_major=False):
    start = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * (cuda.blockDim.x * cuda.blockDim.y)
    step = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
    stop = dim[0] * dim[1]
    for index in range(start, stop, step):
        col = index % dim[1]
        row = index // dim[1]
        if row_major:
            smem[row * ld + col] = matrix[row, col]
        else:
            smem[col * ld + row] = matrix[row, col]


@cuda.jit(device=True, forceinline=True)
def load_to_shared_2d(matrix, smem, dim, row_major=False):
    start = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * (cuda.blockDim.x * cuda.blockDim.y)
    step = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
    stop = dim[0] * dim[1]
    for index in range(start, stop, step):
        col = index % dim[1]
        row = index // dim[1]
        if row_major:
            smem[row, col] = matrix[row, col]
        else:
            smem[col, row] = matrix[row, col]


@cuda.jit(device=True, forceinline=True)
def load_to_shared_1d_float16x2(matrix, smem, dim, ld, row_major=False):
    start = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * (cuda.blockDim.x * cuda.blockDim.y)
    step = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
    stop = dim[0] * dim[1]
    for index in range(start, stop, step):
        col = index % dim[1]
        row = index // dim[1]
        r = matrix[row, 2 * col + 0]
        i = matrix[row, 2 * col + 1]
        if row_major:
            smem[row * ld + col] = float16x2(r, i)
        else:
            smem[col * ld + row] = float16x2(r, i)


@cuda.jit(device=True, forceinline=True)
def store_from_shared_batched(smem, matrix, batch, dim, ld):
    start = cuda.threadIdx.x
    step = cuda.blockDim.x
    stop = dim[0] * dim[1]
    for index in range(start, stop, step):
        col = index % dim[1]
        row = index // dim[1]
        matrix[batch, row, col] = smem[batch * dim[1] * ld + col * ld + row]


@cuda.jit(device=True, forceinline=True)
def store_from_shared(smem, matrix, dim, ld, row_major=False):
    start = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * (cuda.blockDim.x * cuda.blockDim.y)
    step = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
    stop = dim[0] * dim[1]
    for index in range(start, stop, step):
        col = index % dim[1]
        row = index // dim[1]
        if row_major:
            matrix[row, col] = smem[row * ld + col]
        else:
            matrix[row, col] = smem[col * ld + row]


@cuda.jit(device=True, forceinline=True)
def store_from_shared_2d(smem, matrix, dim):
    start = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * (cuda.blockDim.x * cuda.blockDim.y)
    step = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
    stop = dim[0] * dim[1]
    for index in range(start, stop, step):
        col = index % dim[1]
        row = index // dim[1]
        matrix[row, col] = smem[col, row]


@cuda.jit(device=True, forceinline=True)
def store_from_shared_1d_float16x2(smem, matrix, dim, ld):
    start = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.z * (cuda.blockDim.x * cuda.blockDim.y)
    step = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
    stop = dim[0] * dim[1]
    for index in range(start, stop, step):
        col = index % dim[1]
        row = index // dim[1]
        ri = smem[col * ld + row]
        matrix[row, 2 * col + 0] = ri.x
        matrix[row, 2 * col + 1] = ri.y

