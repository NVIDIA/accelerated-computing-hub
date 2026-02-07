# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings import runtime as cudart
import numpy as np
import math

from nvmath.device.common_cuda import ComputeCapability


def random_complex(shape, real_dtype, order="C") -> np.ndarray:
    return random_real(shape, real_dtype, order) + 1.0j * random_real(shape, real_dtype, order)


def random_real(shape, real_dtype, order="C") -> np.ndarray:
    # NOTE: reshape does not guarantee layout if order is provided. So we have
    #   to use copy.
    return np.random.randn(np.prod(shape)).astype(real_dtype).reshape(shape).copy(order=order)


def random_int(shape, int_dtype, order="C"):
    """
    Generate random integers in the range [-2, 2) for signed integers and [0, 4)
    for unsigned integers.
    """
    min_val, max_val = 0, 4
    if issubclass(int_dtype, np.signedinteger):
        min_val, max_val = -2, 2
    # NOTE: reshape does not guarantee layout if order is provided. So we have
    #   to use copy.
    return np.random.randint(min_val, max_val, size=shape, dtype=int_dtype).copy(order=order)


def random(shape, dtype, order=None, arrangement=None):
    assert order is None or arrangement is None, "Specify only one of order or arrangement"
    if arrangement is not None:
        order = "C" if arrangement == "row_major" else "F"
    if order is None:
        order = "C"
    if np.issubdtype(dtype, np.floating):
        return random_real(shape, dtype, order)
    elif np.issubdtype(dtype, np.complexfloating):
        return random_complex(shape, dtype, order)
    elif np.issubdtype(dtype, np.integer):
        return random_int(shape, dtype, order)


def CHECK_CUDART(err):
    if err != cudart.cudaError_t.cudaSuccess:
        err2, str = cudart.cudaGetErrorString(cudart.cudaError_t.cudaSuccess)
        raise RuntimeError(f"CUDArt Error: {str}")


def fft_perf_GFlops(fft_size, batch, time_ms, coef=1.0):
    fft_flops_per_batch = coef * 5.0 * fft_size * math.log2(fft_size)
    return batch * fft_flops_per_batch / (1e-3 * time_ms) / 1e9


def mm_perf_GFlops(size, batch, time_ms, coef=1.0):
    return coef * 2.0 * batch * size[0] * size[1] * size[2] / (1e-3 * time_ms) / 1e9


def fp16x2_to_complex64(data):
    return data[..., ::2] + 1.0j * data[..., 1::2]


def complex64_to_fp16x2(data):
    shape = (*data.shape[:-1], data.shape[-1] * 2)
    output = np.zeros(shape=shape, dtype=np.float16)
    output[..., 0::2] = data.real
    output[..., 1::2] = data.imag
    return output


def device_shared_memory(cc: ComputeCapability) -> int:
    # Source for these chip memory numbers:
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
    match cc.integer:
        case 1200 | 1210:
            return 99 * 1024
        case 900 | 1000 | 1010 | 1030 | 1100:
            return 227 * 1024
        case 890 | 860:
            return 99 * 1024
        case 800 | 870:
            return 163 * 1024
        case 750:
            return 64 * 1024
        case 700 | 720:
            return 96 * 1024
        case _:
            return 48 * 1024
