#include "ach.h"

int main()
{
  int height = 4096;
  int width  = 4096;

  // TODO: Use explicit memory space containers
  thrust::universal_vector<float> prev = ach::init(height, width);
  thrust::universal_vector<float> next(height * width);

  for (int write_step = 0; write_step < 3; write_step++) {
    std::printf("   write step %d\n", write_step);
    ach::store(write_step, height, width, prev);

    for (int compute_step = 0; compute_step < 3; compute_step++) {
      auto begin = std::chrono::high_resolution_clock::now();
      ach::simulate(height, width, prev, next);
      auto end = std::chrono::high_resolution_clock::now();
      auto seconds = std::chrono::duration<double>(end - begin).count();
      std::printf("computed step %d in %g s\n", compute_step, seconds);
      prev.swap(next);
    }
  }
}
