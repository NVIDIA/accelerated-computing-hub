#include "ach.h"

void simulate(int width, int height, const thrust::device_vector<float> &in,
              thrust::device_vector<float> &out, cudaStream_t stream = 0) {
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  cub::DeviceTransform::Transform(
      thrust::make_counting_iterator(0), out.begin(), width * height,
      [=] __host__ __device__(int id) { return ach::compute(id, temp_in); },
      stream);
}

int main() {
  int height = 2048;
  int width = 8192;

  thrust::device_vector<float> d_prev = ach::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::device_vector<float> d_buffer(height * width);
  thrust::host_vector<float> h_prev(height * width);

  const int compute_steps = 750;
  const int write_steps = 3;

  // 1. Create compute and copy streams

  for (int write_step = 0; write_step < write_steps; write_step++) {
    thrust::copy(d_prev.begin(), d_prev.end(), d_buffer.begin());

    // 2. Replace `thrust::copy` with `cudaMemcpyAsync` on copy stream.
    //    Use `thrust::raw_pointer_cast(vec.data())` to get raw pointers from
    //    Thrust containers.
    thrust::copy(d_buffer.begin(), d_buffer.end(), h_prev.begin());

    for (int compute_step = 0; compute_step < compute_steps; compute_step++) {
      // 3. Put `simulate` on compute stream
      simulate(width, height, d_prev, d_next);
      d_prev.swap(d_next);
    }

    // 4. Make sure to synchronize copy stream before reading `h_prev`
    ach::store(write_step, height, width, h_prev);

    // 5. Make sure to synchronize compute stream before next iteration
    cudaDeviceSynchronize();
  }
}
