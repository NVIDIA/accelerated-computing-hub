#include "ach.h"

void simulate(int width,
              int height,
              const thrust::device_vector<float> &in,
                    thrust::device_vector<float> &out)
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  // TODO: replace with an asynchronous call
  thrust::tabulate(out.begin(), out.end(), [=] __device__(int id) {
    return ach::compute(id, temp_in);
  });
}

int main()
{
  int height = 2048;
  int width  = 8192;

  thrust::device_vector<float> d_prev = ach::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::host_vector<float> h_prev(height * width);

  const int compute_steps = 500;
  const int write_steps = 3;

  /* Executing write_steps iterations of
   *  1. Device to Host copy of prev
   *  2. Simulate next on the device
   *  3. Write host prev to disk
   * The goal is to overlap 2. and 3. by using asynchrony
  */
  for (int write_step = 0; write_step < write_steps; write_step++)
  {
    auto step_begin = std::chrono::high_resolution_clock::now();

    // 1. Copying the device prev buffer of previous step in the host buffer
    thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());

    // 2. Executing compute_step iterations to simulate
    // We want this step to now run asynchronously on the GPU
    for (int compute_step = 0; compute_step < compute_steps; compute_step++)
    {
      simulate(width, height, d_prev, d_next);
      d_prev.swap(d_next);
    }

    // TODO: Don't forget to synchronize where it makes senses

    auto write_begin = std::chrono::high_resolution_clock::now();

    // 3. Store host prev to disk
    ach::store(write_step, height, width, h_prev);

    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_seconds = std::chrono::duration<double>(write_end - write_begin).count();

    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_seconds = std::chrono::duration<double>(step_end - step_begin).count();
    std::printf("compute + write %d in %g s\n", write_step, step_seconds);
    std::printf("          write %d in %g s\n", write_step, write_seconds);
  }
}
