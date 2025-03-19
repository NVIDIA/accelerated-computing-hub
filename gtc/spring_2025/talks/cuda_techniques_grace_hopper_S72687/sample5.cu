// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstdlib>

#include <cub/block/block_reduce.cuh>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime_api.h>

constexpr int block_size = 128;
using BlockReduce = cub::BlockReduce<float, block_size>;

__device__ void do_work(float *data, int n, float *output,
                        typename BlockReduce::TempStorage &temp_storage) {
  int tid = threadIdx.x;

  float sum = 0;
  for (int index = tid; index < n; index += blockDim.x) {
    sum += data[index];
  }
  __syncthreads();
  sum = BlockReduce(temp_storage).Sum(sum);

  if (tid == 0)
    *output = sum;
}

__device__ void load_work_queue(
    cuda::atomic_ref<int, cuda::thread_scope_system> work_idx,
    cuda::atomic_ref<int, cuda::thread_scope_system> total_work_items,
    int &smem_work_idx, int &smem_total_work_items) {

  if (threadIdx.x == 0) {
    smem_work_idx = work_idx.load(cuda::memory_order_acquire);
    smem_total_work_items = total_work_items.load(cuda::memory_order_acquire);
  }
  __syncthreads();
}

__global__ void consumer_kernel(float **data_ptrs, float *data_sums,
                                int *work_idx, int *total_work_items,
                                int max_work_items, int elems_per_work_item) {

  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ int _total_work_items;
  __shared__ int _work_idx;

  cuda::atomic_ref<int, cuda::thread_scope_system> a_work_idx(*work_idx);
  cuda::atomic_ref<int, cuda::thread_scope_system> a_total_work_items(
      *total_work_items);

  int tid = threadIdx.x;

  // Loop until all work is produced and consumed
  load_work_queue(a_work_idx, a_total_work_items, _work_idx, _total_work_items);
  while (_total_work_items < max_work_items) {

    // Spin wait until some work is available
    load_work_queue(a_work_idx, a_total_work_items, _work_idx,
                    _total_work_items);
    while (_work_idx >= _total_work_items) {
      load_work_queue(a_work_idx, a_total_work_items, _work_idx,
                      _total_work_items);
      if (_total_work_items >= max_work_items) return;
    }
    __syncthreads();

    load_work_queue(a_work_idx, a_total_work_items, _work_idx,
                    _total_work_items);
    while (_work_idx < _total_work_items) {
      // Get a work item
      if (tid == 0) {
        _work_idx = a_work_idx.fetch_add(1, cuda::memory_order_acq_rel);
      }
      __syncthreads();
      int widx = _work_idx;

      if (widx >= max_work_items)
        return;
      else if (widx >= _total_work_items) {
        // Spin wait until some work is available
        load_work_queue(a_work_idx, a_total_work_items, _work_idx,
                        _total_work_items);
        // Wait for already reserved work to become available
        while (widx >= _total_work_items) {
          load_work_queue(a_work_idx, a_total_work_items, _work_idx,
                          _total_work_items);
        }
      }

      // At this point widx block of work should be ready to go
      // Now get the data pointer
      float *data = data_ptrs[widx];

      // Do work
      do_work(data, elems_per_work_item, &data_sums[widx], temp_storage);

      load_work_queue(a_work_idx, a_total_work_items, _work_idx,
                      _total_work_items);
    }
  }
}

int main() {
  int device_id;
  cudaGetDevice(&device_id);

  int isHostNativeAtomicSupported;
  cudaDeviceGetAttribute(&isHostNativeAtomicSupported,
                         cudaDevAttrHostNativeAtomicSupported, device_id);
  printf("cudaDevAttrHostNativeAtomicSupported: %d\n",
         isHostNativeAtomicSupported);

  int max_work_items = 1024;
  int elems_per_work_item = 1024;
  // Allocate all the work buffers
  float **data_ptrs = (float **)malloc(max_work_items * sizeof(float *));
  float *data_sums = (float *)malloc(max_work_items * sizeof(float));
  for (int i = 0; i < max_work_items; ++i) {
    data_ptrs[i] = (float *)malloc(elems_per_work_item * sizeof(float));
    data_sums[i] = 0.f;
  }

  // Set up the work queue
  int *total_work_items = (int *)malloc(sizeof(int));
  int *work_idx = (int *)malloc(sizeof(int));
  *total_work_items = 0;
  *work_idx = 0;

  cuda::atomic_ref<int, cuda::thread_scope_system> a_work_idx(*work_idx);
  cuda::atomic_ref<int, cuda::thread_scope_system> a_total_work_items(
      *total_work_items);

  // Launch consumer kernel on the GPU
  int grid = 128;
  consumer_kernel<<<grid, block_size>>>(data_ptrs, data_sums, work_idx,
                                        total_work_items, max_work_items,
                                        elems_per_work_item);

  // Start producing work for the kernel to consume
  for (int i = 0; i < max_work_items; ++i) {
    for (int j = 0; j < elems_per_work_item; ++j)
      data_ptrs[i][j] = 1.f;

    // Add this to the queue
    a_total_work_items.fetch_add(1, cuda::memory_order_acq_rel);
  }
  printf("CPU finished submitting %d work items\n", a_total_work_items.load());
  cudaDeviceSynchronize();

  for (int i = 0; i < max_work_items; i++) {
    printf("Work item %d: computed sum %g, expected sum %g) \n", i,
    data_sums[i], (float)elems_per_work_item);
  }

  free(data_sums);
  free(total_work_items);
  free(work_idx);
  for (int i = 0; i < max_work_items; ++i)
    free(data_ptrs[i]);
  free(data_ptrs);
  return 0;
}
