#include "ach.cuh"

__global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                 cuda::std::span<int> histogram) {
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int bin = static_cast<int>(temperatures[cell] / 10);

  cuda::std::atomic_ref<int> ref(histogram[bin]);
  ref.fetch_add(1);
}

void histogram(cuda::std::span<float> temperatures,
               cuda::std::span<int> histogram, cudaStream_t stream) {
  int block_size = 256;
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(temperatures,
                                                         histogram);
}
