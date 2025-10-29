#pragma once

#include <cub/device/device_for.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/universal_vector.h>

#include <cuda/std/mdspan>

#include <fstream>

#if CUB_VERSION >= 200800
#include <cub/device/device_transform.cuh>
#else
// TODO Will be part of CTK 12.8
namespace cub {
struct DeviceTransform {
  template <class InputIt, class OutputIt, class UnaryOperation>
  static cudaError_t Transform(InputIt in, OutputIt out, int num_items,
                               UnaryOperation op, cudaStream_t stream = 0) {
    return cub::DeviceFor::Bulk(num_items,
                                [=] __device__(int i) { out[i] = op(in[i]); });
  }
};
} // namespace cub

// TODO Will be part of CTK 12.8
namespace cuda {
int ceil_div(int a, int b) { return (a + b - 1) / b; }
} // namespace cuda

namespace thrust {

namespace system {
namespace cuda {
using universal_host_pinned_memory_resource = detail::pinned_memory_resource;
}
} // namespace system

template <typename T>
using universal_host_pinned_allocator =
    thrust::mr::stateless_resource_allocator<
        T, thrust::system::cuda::universal_host_pinned_memory_resource>;

// Should be part of CTK 12.8
template <typename T>
using universal_host_pinned_vector =
    thrust::detail::vector_base<T, universal_host_pinned_allocator<T>>;

} // namespace thrust
#endif

namespace ach {

using temperature_grid_f =
    cuda::std::mdspan<float, cuda::std::dextents<int, 2>>;

using temperature_grid_d =
    cuda::std::mdspan<double, cuda::std::dextents<int, 2>>;

__host__ __device__ float compute(int cell_id, temperature_grid_f temp) {
  int height = temp.extent(0);
  int width = temp.extent(1);

  int column = cell_id % width;
  int row = cell_id / width;

  if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
    float d2tdx2 =
        temp(row, column - 1) - 2 * temp(row, column) + temp(row, column + 1);
    float d2tdy2 =
        temp(row - 1, column) - 2 * temp(row, column) + temp(row + 1, column);

    return temp(row, column) + 0.2f * (d2tdx2 + d2tdy2);
  } else {
    return temp(row, column);
  }
}

__global__ void grid_kernel(temperature_grid_f in, float *out) {
  int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  int number_of_threads = blockDim.x * gridDim.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) {
    out[id] = ach::compute(id, in);
  }
}

void simulate(ach::temperature_grid_f temp_in, float *temp_out,
              cudaStream_t stream) {
  int block_size = 1024;
  int grid_size = cuda::ceil_div(temp_in.size(), block_size);

  grid_kernel<<<grid_size, block_size, 0, stream>>>(temp_in, temp_out);
}

void store(int step, int height, int width,
           const thrust::universal_host_pinned_vector<float> &data) {}

} // namespace ach

void symmetry_check(ach::temperature_grid_f temp, cudaStream_t stream);

void simulate(ach::temperature_grid_f temp, float *temp_out,
              cudaStream_t stream) {
  symmetry_check(temp, stream);
  ach::simulate(temp, temp_out, stream);
}

int main() {
  int height = 1024;
  int width = 5000;

  cudaStream_t compute_stream;
  cudaStreamCreate(&compute_stream);

  // Trying to silence symmetry check error
  {
    thrust::device_vector<float> d_prev((height + 1) * width);
    thrust::device_vector<float> d_next((height + 1) * width);
  }

  thrust::device_vector<float> d_prev(height * width);
  // thrust::fill_n(d_prev.begin(), width, 90.0f);
  // thrust::fill_n(d_prev.begin() + width * (height - 1), width, 90.0f);
  thrust::device_vector<float> d_next(height * width);

  auto step_begin = std::chrono::high_resolution_clock::now();
  for (int compute_step = 0; compute_step < 10; compute_step++) {
    ach::temperature_grid_f temp_in(thrust::raw_pointer_cast(d_prev.data()),
                                    height, width);
    float *temp_out = thrust::raw_pointer_cast(d_next.data());
    simulate(temp_in, temp_out, compute_stream);
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    d_prev.swap(d_next);
  }
  cudaStreamSynchronize(compute_stream);
  auto step_end = std::chrono::high_resolution_clock::now();
  auto step_seconds =
      std::chrono::duration<double>(step_end - step_begin).count();

  std::printf("compute in %g s\n", step_seconds);

  cudaStreamDestroy(compute_stream);
}