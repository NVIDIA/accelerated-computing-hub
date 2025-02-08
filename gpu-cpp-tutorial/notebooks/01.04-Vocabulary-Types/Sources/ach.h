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
#include <thrust/iterator/transform_iterator.h>
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

static 
thrust::universal_vector<float> init(int height, int width)
{
  const float low  = 15.0;
  const float high = 90.0;
  thrust::universal_vector<float> data(height * width, low);
  thrust::fill(thrust::device, data.begin(), data.begin() + width, high);
  thrust::fill(thrust::device, data.end() - width, data.end(), high);
  return data;
}

static
void store(int step, int height, int width, const thrust::universal_vector<float> &data)
{
  std::ofstream file("/tmp/heat_" + std::to_string(step) + ".bin", std::ios::binary);
  file.write(reinterpret_cast<const char*>(&height), sizeof(int));
  file.write(reinterpret_cast<const char*>(&width), sizeof(int));
  file.write(reinterpret_cast<const char *>(data.data().get()), height * width * sizeof(float));
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

void simulate(int height, int width, 
              const thrust::universal_vector<float> &in,
                    thrust::universal_vector<float> &out);

int main() 
{
  int height = 256;
  int width = 1024;

  thrust::universal_vector<float> prev = ach::init(height, width);
  thrust::universal_vector<float> next(height * width);

  for (int write_step = 0; write_step < 100; write_step++) 
  {
    for (int compute_step = 0; compute_step < 100; compute_step++) 
    {
      simulate(height, width, prev, next);
      next.swap(prev);
    }

    ach::store(write_step, height, width, prev);
  }
}
