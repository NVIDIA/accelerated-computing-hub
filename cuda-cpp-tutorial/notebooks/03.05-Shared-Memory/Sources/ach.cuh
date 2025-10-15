#pragma once

#include <cub/device/device_for.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/universal_vector.h>

#include <cuda/atomic>
#include <cuda/std/atomic>
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

template <class ContainerT> void store(int step, int bins, ContainerT &data) {
  std::ofstream file("/tmp/hist_" + std::to_string(step) + ".bin",
                     std::ios::binary);
  file.write(reinterpret_cast<const char *>(&bins), sizeof(int));
  file.write(
      reinterpret_cast<const char *>(thrust::raw_pointer_cast(data.data())),
      bins * sizeof(int));
}

} // namespace ach

void histogram(cuda::std::span<float> temperatures,
               cuda::std::span<int> block_histograms,
               cuda::std::span<int> histogram, cudaStream_t stream);

int main() {
  unsigned height = 1024;
  unsigned width = 4096;

  cudaStream_t compute_stream;
  cudaStreamCreate(&compute_stream);

  thrust::device_vector<int> d_block_histograms(height * width * 10, 0);
  thrust::device_vector<int> d_histogram(10);
  thrust::host_vector<int> h_histogram(10);

  float low = 0.0f;
  float high = 99.0f;
  thrust::host_vector<float> h_prev(height * width, low);
  thrust::device_vector<float> d_prev(height * width, low);
  thrust::fill_n(d_prev.begin(), width, high);
  thrust::fill_n(d_prev.begin() + width * (height - 1), width, high);
  thrust::device_vector<float> d_next(height * width);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  for (int write_step = 0; write_step < 100; write_step++) {
    h_prev = d_prev;
    thrust::fill(d_histogram.begin(), d_histogram.end(), 0);
    thrust::fill(d_block_histograms.begin(), d_block_histograms.end(), 0);
    cudaEventRecord(begin, compute_stream);
    histogram(
        cuda::std::span{thrust::raw_pointer_cast(d_prev.data()), d_prev.size()},
        cuda::std::span{thrust::raw_pointer_cast(d_block_histograms.data()),
                        d_block_histograms.size()},
        cuda::std::span{thrust::raw_pointer_cast(d_histogram.data()),
                        d_histogram.size()},
        compute_stream);
    cudaEventRecord(end, compute_stream);
    cudaEventSynchronize(end);
    float ms{};
    cudaEventElapsedTime(&ms, begin, end);
    std::printf("histogram took %f ms\n", ms);
    h_histogram = d_histogram;

    if (thrust::reduce(h_histogram.begin(), h_histogram.end()) !=
        height * width) {
      std::printf("Error: sum of bins is not equal to number of cells\n");
    }

    ach::store(write_step, 10, h_histogram);
    ach::store(write_step, height, width, h_prev);
    for (int compute_step = 0; compute_step < 1200; compute_step++) {
      ach::simulate(
          ach::temperature_grid_f{thrust::raw_pointer_cast(d_prev.data()),
                                  height, width},
          thrust::raw_pointer_cast(d_next.data()), compute_stream);
      d_prev.swap(d_next);
    }
  }
  cudaStreamSynchronize(compute_stream);

  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaStreamDestroy(compute_stream);
}
