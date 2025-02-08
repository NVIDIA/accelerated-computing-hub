#pragma once

#include <cstdio>

#include <nv/target>

#include <cstdint> // CHAR_BIT

#include <thrust/fill.h>
#include <thrust/tabulate.h>
#include <thrust/universal_vector.h>

#include <cstdio>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/universal_vector.h>

#include <cuda/std/mdspan>

#include <fstream>

namespace ach {

static __host__ __device__ bool is_executed_on_gpu() {
  NV_IF_TARGET(NV_IS_HOST, (return false;));
  return true;
}

static __host__ __device__ const char *execution_space() {
  return is_executed_on_gpu() ? "GPU" : "CPU";
}

static double max_bandwidth() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  const std::size_t mem_freq =
      static_cast<std::size_t>(prop.memoryClockRate) * 1000; // kHz -> Hz
  const int bus_width = prop.memoryBusWidth;
  const std::size_t bytes_per_second = 2 * mem_freq * bus_width / CHAR_BIT;
  return static_cast<double>(bytes_per_second) / 1024 / 1024 /
         1024; // B/s -> GB/s
}

__host__ __device__ void I_expect(const char *expected) {
  std::printf("expect %s; runs on %s;\n", expected, execution_space());
}

thrust::universal_vector<float> init(int height, int width) {
  const float low = 15.0;
  const float high = 90.0;
  thrust::universal_vector<float> temp(height * width, low);
  thrust::fill(thrust::device, temp.begin(), temp.begin() + width, high);
  return temp;
}

void report(int height, int width,
            const thrust::universal_vector<float> &temp,
            const thrust::universal_vector<float> &sums,
            std::chrono::time_point<std::chrono::high_resolution_clock> begin,
            std::chrono::time_point<std::chrono::high_resolution_clock> end) {
  const double seconds = std::chrono::duration<double>(end - begin).count();
  const double gigabytes =
      static_cast<double>(temp.size() * sizeof(float)) / 1024 / 1024 / 1024;
  const double throughput = gigabytes / seconds;

  std::printf("computed in %g s\n", seconds);
  std::printf("achieved throughput: %g GB/s\n", throughput);
  std::printf("maximal bandwidth: %g GB/s\n", max_bandwidth());

  for (int row_id = 0; row_id < 3; row_id++) {
    float first = temp[row_id * width];
    float second = temp[row_id * width + 1];
    float last = temp[(row_id + 1) * width - 1];
    float sum = sums[row_id];
    std::printf("row %d: { %g, %g, ..., %g } = %g\n", row_id, first, second,
                last, sum);
  }
}

} // namespace ach

namespace heat {
inline thrust::universal_vector<float> generate_random_data(int height,
                                                            int width) {
  const float low = 15.0;
  const float high = 90.0;
  thrust::universal_vector<float> data(height * width, low);
  thrust::fill(thrust::device, data.begin(), data.begin() + width, high);
  thrust::fill(thrust::device, data.end() - width, data.end(), high);
  return data;
}

template <class ContainerT>
void simulate(int height, int width, const ContainerT &in, ContainerT &out) {
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);

  thrust::tabulate(
      thrust::device, out.begin(), out.end(), [=] __host__ __device__(int id) {
        const int column = id % width;
        const int row = id / width;

        // loop over all points in domain (except boundary)
        if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
          // evaluate derivatives
          float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) +
                         temp_in(row, column + 1);
          float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) +
                         temp_in(row + 1, column);

          // update temperatures
          return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
        } else {
          return temp_in(row, column);
        }
      });
}
} // namespace heat

thrust::universal_vector<float> row_temperatures(
    int height, int width,
    thrust::universal_vector<int>& row_ids,
    thrust::universal_vector<float>& temp);

int main() 
{
    int width = 16777216;
    int height = 16;

    thrust::universal_vector<float> temp = ach::init(height, width);
    thrust::universal_vector<int> row_ids(height * width);
    thrust::tabulate(row_ids.begin(), row_ids.end(), [=]__host__ __device__(int i) { return i / width; });

    auto begin = std::chrono::high_resolution_clock::now();
    thrust::universal_vector<float> sums = row_temperatures(height, width, row_ids, temp);
    auto end = std::chrono::high_resolution_clock::now();

    ach::report(height, width, temp, sums, begin, end);
}
