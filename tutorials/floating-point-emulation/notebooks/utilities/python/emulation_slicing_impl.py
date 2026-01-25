import numpy as np
import time
from enum import IntEnum

from numba import cuda
from cuda import coop

from emulation_utils import *

class slice_matrix(IntEnum):
    a = 0
    b = 1

def get_slice_kernel(SliceMatrix, Slices, BlockSize=64):
    uint8_width = get_width(np.uint8)
    int8_width = get_width(np.int8)

    @cuda.jit(device=True)
    def slices_from_fp64(slices, val: np.float64, exponent_shift: np.int32):
        normalization_factor = float(2**52)

        skip_slices = 0
        r           = 0
        reg_pack    = 0

        r0_exponent = extract_exponent(val)
        denorm_compensation = 0

        if r0_exponent == 0:
            if val == 0.0:
                skip_slices = Slices
                r = 0
            else:
                # round to nearest is the default behaviour on CPU
                val                 = (val * normalization_factor)
                r0_exponent         = extract_exponent(val)
                denorm_compensation = -52

        exp = r0_exponent + exponent_shift + denorm_compensation - BIAS
        exp += (Slices - 1) * uint8_width # Use all 8 bits

        # Adjust casting range
        extra_width = (exp + 1) - 63
        extra_width = extra_width if extra_width > 0 else 0
        skip_slices = div_up(extra_width, uint8_width)
        exp -= skip_slices * uint8_width

        if exp < 0:
            r = 0
        else:
            val = copy_with_exponent(val, exp + BIAS)
            r = np.int64(val)

        for _i in range(0, Slices):
            i = Slices - 1 - _i

            if _i < skip_slices:
                reg_pack = np.uint8(0)
            else:
                reg_pack  = np.uint8(r)
                slices[i] = np.int8(reg_pack)
                r         = np.int64((r >> uint8_width) + (reg_pack >> int8_width))
    
    @cuda.jit(launch_bounds=(BlockSize, 2))
    def slice_kernel(in_tensor, shift_tensor, out_tensor, reduction_dim_size):
        tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

        row_idx = tid // reduction_dim_size
        col_idx = tid %  reduction_dim_size

        if SliceMatrix == slice_matrix.a:
            shift_idx = row_idx
        else:
            shift_idx = col_idx

        slices = cuda.local.array(shape=(Slices,), dtype=np.int8)
        slices_from_fp64(slices, in_tensor[row_idx, col_idx], shift_tensor[shift_idx])

        for elem in range(0, Slices):
            out_tensor[row_idx, col_idx, elem] = slices[elem]

    return slice_kernel

def get_max_reduce_kernel(SliceMatrix, BlockSize=64):

    @cuda.jit(device=True)
    def max_op(a, b):
        return a if a > b else b

    block_reduce = coop.block.reduce(np.float64, BlockSize, max_op)

    @cuda.jit(launch_bounds=(BlockSize, 2), link=block_reduce.files)
    def max_reduce_kernel(in_tensor, out_tensor):
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x

        global_tile = in_tensor[bid, :] if SliceMatrix == slice_matrix.a else in_tensor[:, bid]

        local_max = 0
        for i in range(tid, len(global_tile), BlockSize):
            local_max = max(local_max, abs(global_tile[i]))

        # TODO(cbrower) check if we can reduce with smem and how this reduction happens
        local_max = block_reduce(local_max)
        
        if tid == 0:
            out_tensor[bid] = max_to_exponent_shift(local_max)
    
    return max_reduce_kernel
