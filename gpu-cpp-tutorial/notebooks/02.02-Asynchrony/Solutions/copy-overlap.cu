#include "ach.h"

void simulate(int width, int height, const thrust::device_vector<float> &in,
              thrust::device_vector<float> &out, cudaStream_t stream) {
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  cub::DeviceTransform::Transform(
      thrust::make_counting_iterator(0), out.begin(), width * height,
      [=] __host__ __device__(int id) { return ach::compute(id, temp_in); },
      stream);
}

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
  thrust::universal_host_pinned_vector<float> h_temp(height * width);

  int compute_steps = 500;
  int write_steps = 100;
  for (int write_step = 0; write_step < write_steps; write_step++) {
    auto step_begin = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer.data()),
                    thrust::raw_pointer_cast(d_prev.data()),
                    height * width * sizeof(float), cudaMemcpyDeviceToDevice,
                    compute_stream);
    cudaStreamSynchronize(compute_stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(h_temp.data()),
                    thrust::raw_pointer_cast(d_buffer.data()),
                    height * width * sizeof(float), cudaMemcpyDeviceToHost,
                    copy_stream);

    for (int compute_step = 0; compute_step < compute_steps; compute_step++) {
      simulate(width, height, d_prev, d_next, compute_stream);
      d_prev.swap(d_next);
    }

    cudaStreamSynchronize(copy_stream);

    auto write_begin = std::chrono::high_resolution_clock::now();
    ach::store(write_step, height, width, h_temp);
    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_seconds =
        std::chrono::duration<double>(write_end - write_begin).count();

    cudaStreamSynchronize(compute_stream);
    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_seconds =
        std::chrono::duration<double>(step_end - step_begin).count();
    std::printf("compute + write %d in %g s\n", write_step, step_seconds);
    std::printf("          write %d in %g s\n", write_step, write_seconds);
  }

  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(copy_stream);
}
