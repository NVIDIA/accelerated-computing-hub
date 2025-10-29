#include "ach.h"

const int number_of_threads = 2048;

__global__ void block_kernel(ach::temperature_grid_f in, float *out) {
  int thread_index = threadIdx.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) {
    out[id] = ach::compute(id, in);
  }
}

void simulate(ach::temperature_grid_f temp_in, float *temp_out,
              cudaStream_t stream) {
  block_kernel<<<1, number_of_threads, 0, stream>>>(temp_in, temp_out);
}
