#include "ach.h"

int main() {
  int height = 2048;
  int width = 8192;

  cudaStream_t compute_stream;
  cudaStreamCreate(&compute_stream);

  cudaStream_t copy_stream;
  cudaStreamCreate(&copy_stream);

  thrust::device_vector<float> d_prev = ach::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::device_vector<float> d_buffer(height * width);

  thrust::universal_host_pinned_vector<float> h_prev(height * width);

  const int compute_steps = 750;
  const int write_steps = 3;
  for (int write_step = 0; write_step < write_steps; write_step++) {
    cudaMemcpy(thrust::raw_pointer_cast(d_buffer.data()),
               thrust::raw_pointer_cast(d_prev.data()),
               height * width * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(thrust::raw_pointer_cast(h_prev.data()),
                    thrust::raw_pointer_cast(d_buffer.data()),
                    height * width * sizeof(float), cudaMemcpyDeviceToHost,
                    copy_stream);

    for (int compute_step = 0; compute_step < compute_steps; compute_step++) {
      ach::simulate(width, height, d_prev, d_next, compute_stream);
      d_prev.swap(d_next);
    }

    cudaStreamSynchronize(copy_stream);
    ach::store(write_step, height, width, h_prev);

    cudaStreamSynchronize(compute_stream);
  }

  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(copy_stream);
}