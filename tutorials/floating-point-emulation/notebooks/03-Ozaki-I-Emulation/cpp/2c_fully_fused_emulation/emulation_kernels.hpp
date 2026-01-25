#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#include <cublasdx.hpp>
using namespace cublasdx;

#include <cub/block/block_reduce.cuh>

#include <tutorial_helpers.hpp>
#include "slicing.hpp"

enum class slice_matrix
{
    a,
    b
};

template<int BlockSize, slice_matrix SliceMatrix, class InTensor, class OutTensor>
__launch_bounds__(BlockSize, 2) __global__ void max_reduce_kernel(InTensor in_tensor, OutTensor out_tensor) {
    using datatype    = tutorial::tensor_value_type_t<InTensor>;
    using BlockReduce = cub::BlockReduce<datatype, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const auto [tile_size_x, tile_size_y] = in_tensor.layout().shape();
    auto tid                              = threadIdx.x;
    auto bid                              = blockIdx.x;

    // Assume that tensor is reduced along the last dimension
    auto const row_index = tutorial::conditional_return<SliceMatrix == slice_matrix::a>(bid, cublasdx::slice);
    auto const col_index = tutorial::conditional_return<SliceMatrix == slice_matrix::a>(cublasdx::slice, bid);

    auto global_tile = in_tensor(row_index, col_index);

    // 1. Find local maximum absolute value for this thread
    double local_max = 0;

    auto const length = (SliceMatrix == slice_matrix::a) ? tile_size_y : tile_size_x;
    for (auto i = tid; i < length; i += BlockSize) {
        local_max = cuda::std::max<double>(local_max, cuda::std::abs(global_tile(i)));
    }

    // 2. Compute block-wide reduction to find maximum across all threads
    __syncthreads();
    const double block_max = BlockReduce(temp_storage).Reduce(local_max, [](const auto& a, const auto& b) {
        return cuda::std::max<double>(a, b);
    });

    // 3. Convert maximum value to exponent shift and store to global memory
    // This shift determines the scaling factor for slicing this row/column
    if (tid == 0) {
        out_tensor(bid) = max_to_exponent_shift(block_max);
    }
}


template<int BlockSize, int Slices, slice_matrix SliceMatrix, class InTensor, class ShiftTensor, class OutTensor>
__launch_bounds__(BlockSize, 2) __global__
    void slice_kernel(InTensor in_tensor, ShiftTensor shift_tensor, OutTensor out_tensor, int32_t reduction_dim_size) {
    using in_datatype  = tutorial::tensor_value_type_t<InTensor>;
    using out_datatype = tutorial::tensor_value_type_t<OutTensor>;

    const auto tid = threadIdx.x + blockIdx.x * BlockSize;

    // Calculate which matrix element this thread processes
    auto slow_idx = tid / reduction_dim_size;
    auto fast_idx = tid % reduction_dim_size;

    auto const row_idx = (SliceMatrix == slice_matrix::a) ? slow_idx : fast_idx;
    auto const col_idx = (SliceMatrix == slice_matrix::a) ? fast_idx : slow_idx;

    // Decompose the double precision value into multiple int8_t slices
    // using the appropriate scaling factor for this row/column
    const cuda::std::array slices =
        slices_from_fp64<out_datatype, Slices>(in_tensor(row_idx, col_idx), shift_tensor(slow_idx));

// Store all slices for this matrix element
#pragma unroll
    for (int elem = 0; elem < Slices; ++elem) {
        out_tensor(row_idx, col_idx, elem) = slices[elem];
    }
}

#include "fused_kernel.hpp.inc"

