import ctypes
import math

import numpy as np

from numba import cuda
from enum import IntEnum

BIAS = 1023

class MatrixHalf(IntEnum):
    lower = 0
    upper = 1

@cuda.jit(device=True, forceinline=True)
def extract_sign(x:np.float64) -> np.int32:
    SIGN_MASK = 0x8000000000000000
    return np.int32(x.view(np.int64) & SIGN_MASK)

@cuda.jit(device=True, forceinline=True)
def copy_with_sign(x:np.float64, sign:np.int32) -> np.float64:
    SIGN_MASK = 0x8000000000000000
    bits = (x.view(np.int64) & ~SIGN_MASK) | (np.int64(sign) << 63)
    return bits.view(np.float64)

@cuda.jit(device=True, forceinline=True)
def extract_exponent(x:np.float64) -> np.int32:
    EXPONENT_MASK = 0x7FF0000000000000
    EXPONENT_SHIFT = 52
    return np.int32((x.view(np.int64) & EXPONENT_MASK) >> EXPONENT_SHIFT)

@cuda.jit(device=True, forceinline=True)
def copy_with_exponent(x:np.float64, exponent:np.int32) -> np.float64:
    EXPONENT_MASK = 0x7FF0000000000000
    EXPONENT_SHIFT = 52
    bits = (x.view(np.int64) & ~EXPONENT_MASK) | (np.int64(exponent) << EXPONENT_SHIFT)
    return bits.view(np.float64)

@cuda.jit(device=True, forceinline=True)
def extract_mantissa_hi(x:np.float64) -> np.int32:
    MANTISSA_HI_MASK = 0x000FFFFF00000000
    return np.int32((x.view(np.int64) & MANTISSA_HI_MASK) >> 32)

@cuda.jit(device=True, forceinline=True)
def copy_with_mantissa_hi(x:np.float64, mantissa_hi:np.int32) -> np.float64:
    MANTISSA_HI_MASK = 0x000FFFFF00000000
    bits = (x.view(np.int64) & ~MANTISSA_HI_MASK) | (np.int64(mantissa_hi) << 32)
    return bits.view(np.float64)

@cuda.jit(device=True, forceinline=True)
def extract_mantissa_lo(x:np.float64) -> np.int32:
    MANTISSA_LO_MASK = 0x00000000FFFFFFFF
    return np.int32(x.view(np.int64) & MANTISSA_LO_MASK)

@cuda.jit(device=True, forceinline=True)
def copy_with_mantissa_lo(x:np.float64, mantissa_lo:np.int32) -> np.float64:
    MANTISSA_LO_MASK = 0x00000000FFFFFFFF
    bits = (x.view(np.int64) & ~MANTISSA_LO_MASK) | (np.int64(mantissa_lo))
    return bits.view(np.float64)

def get_width(dtype: np.dtype):
    if dtype == np.int8:
        return 7
    elif dtype == np.uint8:
        return 8

@cuda.jit(device=True, forceinline=True)
def div_up(x, y):
    return (x + y - 1) // y

# Host version - works with dtype objects
def max_exponent(dtype: np.dtype):
    if dtype == np.int8:
        return 7
    elif dtype == np.uint8:
        return 8

# Device version - uses compile-time constant
@cuda.jit(device=True, forceinline=True)
def max_exponent_int8():
    return 7

@cuda.jit(device=True, forceinline=True)
def max_exponent_uint8():
    return 8

@cuda.jit(device=True, forceinline=True)
def get_exponent(x: np.float64):
    em_exponent = extract_exponent(x) + 1 - BIAS

    # Check if top 6 bits of mantissa_hi are set (63 << 14 = 0xFC000)
    if extract_mantissa_hi(x) & (63 << 14) == (63 << 14):
        em_exponent += 1

    return em_exponent

@cuda.jit(device=True, forceinline=True)
def max_to_exponent_shift(row_col_max: np.float64):
    scale_max_exponent = max_exponent_int8()

    return scale_max_exponent - get_exponent(row_col_max)

@cuda.jit(device=True, forceinline=True)
def epilogue_ldexp(em: np.float64, exp: int) -> np.float64:
    exp_max = BIAS - 1
    previous_exp_biased = extract_exponent(em)
    
    if 0 < previous_exp_biased and 0 < previous_exp_biased + exp and previous_exp_biased + exp <= exp_max + BIAS:
        em = copy_with_exponent(em, previous_exp_biased + exp)
        return em
    
    # Use math.ldexp for cases outside fast path
    return math.ldexp(em, exp)
