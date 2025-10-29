#include "ach.h"

int main() {
  int height = 4096;
  int width = 4096;

  thrust::device_vector<float> d_prev = ach::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::host_vector<float> h_prev(height * width);

  for (int write_step = 0; write_step < 3; write_step++) {
    std::printf("   write step %d\n", write_step);
    thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
    ach::store(write_step, height, width, h_prev);

    for (int compute_step = 0; compute_step < 3; compute_step++) {
      auto begin = std::chrono::high_resolution_clock::now();
      ach::simulate(height, width, d_prev, d_next);
      auto end = std::chrono::high_resolution_clock::now();
      auto seconds = std::chrono::duration<double>(end - begin).count();
      std::printf("computed step %d in %g s\n", compute_step, seconds);
      d_prev.swap(d_next);
    }
  }
}
