#include "ach.cuh"

constexpr float bin_width = 10;

__global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                 cuda::std::span<int> histogram) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < temperatures.size()) {
    int bin = static_cast<int>(temperatures[cell] / bin_width);

    // fix data race in incrementing histogram bins by using
    // `cuda::std::atomic_ref`
    int old_count = histogram[bin];
    int new_count = old_count + 1;
    histogram[bin] = new_count;
  }
}

void histogram(cuda::std::span<float> temperatures,
               cuda::std::span<int> histogram, cudaStream_t stream) {
  int block_size = 256;
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(temperatures,
                                                         histogram);
}
