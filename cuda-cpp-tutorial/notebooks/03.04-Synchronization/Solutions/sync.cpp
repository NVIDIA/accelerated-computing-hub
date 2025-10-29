#include "ach.cuh"

constexpr float bin_width = 10;

__global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                 cuda::std::span<int> block_histograms,
                                 cuda::std::span<int> histogram) {
  cuda::std::span<int> block_histogram =
      block_histograms.subspan(blockIdx.x * histogram.size(), histogram.size());

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int bin = static_cast<int>(temperatures[cell] / bin_width);

  cuda::atomic_ref<int, cuda::thread_scope_block> block_ref(
      block_histogram[bin]);
  block_ref.fetch_add(1);
  __syncthreads();

  if (threadIdx.x < histogram.size()) {
    cuda::atomic_ref<int, cuda::thread_scope_device> ref(
        histogram[threadIdx.x]);
    ref.fetch_add(block_histogram[threadIdx.x]);
  }
}

void histogram(cuda::std::span<float> temperatures,
               cuda::std::span<int> block_histograms,
               cuda::std::span<int> histogram, cudaStream_t stream) {
  int block_size = 256;
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(
      temperatures, block_histograms, histogram);
}