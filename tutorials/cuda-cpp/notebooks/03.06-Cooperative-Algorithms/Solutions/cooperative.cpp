#include "ach.cuh"

constexpr int block_size = 256;
constexpr int items_per_thread = 1;
constexpr int num_bins = 10;
constexpr float bin_width = 10;

__global__ void histogram_kernel(cuda::std::span<float> temperatures,
                                 cuda::std::span<int> histogram) {
  __shared__ int block_histogram[num_bins];

  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  int bins[items_per_thread] = {
      static_cast<int>(temperatures[cell] / bin_width)};

  using histogram_t =
      cub::BlockHistogram<int, block_size, items_per_thread, num_bins,
                          cub::BlockHistogramAlgorithm::BLOCK_HISTO_ATOMIC>;
  __shared__ typename histogram_t::TempStorage temp_storage;
  histogram_t(temp_storage).Histogram(bins, block_histogram);

  __syncthreads();
  if (threadIdx.x < num_bins) {
    cuda::atomic_ref<int, cuda::thread_scope_device> ref(
        histogram[threadIdx.x]);
    ref.fetch_add(block_histogram[threadIdx.x]);
  }
}

void histogram(cuda::std::span<float> temperatures,
               cuda::std::span<int> histogram, cudaStream_t stream) {
  int grid_size = cuda::ceil_div(temperatures.size(), block_size);
  histogram_kernel<<<grid_size, block_size, 0, stream>>>(temperatures,
                                                         histogram);
}
