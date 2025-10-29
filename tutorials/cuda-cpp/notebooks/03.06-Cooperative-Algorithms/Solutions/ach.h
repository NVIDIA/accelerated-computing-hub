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

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_histogram.cuh>

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

void simulate(ach::temperature_grid_f temp_in, float *temp_out) {
  int block_size = 1024;
  int grid_size = cuda::ceil_div(temp_in.size(), block_size);

  grid_kernel<<<grid_size, block_size>>>(temp_in, temp_out);
}

template <class ContainerT>
void store(int step, int height, int width, ContainerT &data) {
  std::ofstream file("/tmp/heat_" + std::to_string(step) + ".bin",
                     std::ios::binary);
  file.write(reinterpret_cast<const char *>(&height), sizeof(int));
  file.write(reinterpret_cast<const char *>(&width), sizeof(int));
  file.write(
      reinterpret_cast<const char *>(thrust::raw_pointer_cast(data.data())),
      height * width * sizeof(float));
}

thrust::universal_vector<float> init(int height, int width) {
  const float low = 1.0;
  const float high = 250.0;
  thrust::universal_vector<float> data(height * width, low);
  thrust::fill(thrust::device, data.begin(), data.begin() + width, high);
  thrust::fill(thrust::device, data.end() - width, data.end(), high);
  return data;
}

} // namespace ach

void coarse(ach::temperature_grid_f fine, ach::temperature_grid_f coarse);

int main() {
  int fine_height = 1024;
  int fine_width = 4096;
  int coarse_height = fine_height / tile_size;
  int coarse_width = fine_width / tile_size;

  thrust::universal_vector<float> prev = ach::init(fine_height, fine_width);
  thrust::universal_vector<float> next(fine_height * fine_width);
  thrust::universal_vector<float> coarse_data(coarse_height * coarse_width);
  ach::temperature_grid_f coarse_mdspan(thrust::raw_pointer_cast(coarse_data.data()), coarse_height, coarse_width);

  int write_step = 0;
  int steps = 50000;
  int write_every = steps / 100;
  for (int compute_step = 0; compute_step < steps; compute_step++) {
    ach::temperature_grid_f temp_in(thrust::raw_pointer_cast(prev.data()), fine_height, fine_width);
    ach::simulate(temp_in, thrust::raw_pointer_cast(next.data()));
    prev.swap(next);

    if (compute_step % write_every == 0) {
      ach::temperature_grid_f fine(thrust::raw_pointer_cast(prev.data()), fine_height, fine_width);
      coarse(fine, coarse_mdspan);
      cudaDeviceSynchronize();
      ach::store(write_step++, coarse_height, coarse_width, coarse_data);
    }
  }
}
