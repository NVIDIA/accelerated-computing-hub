#include "ach.h"

__global__ void single_thread_kernel(ach::temperature_grid_f in, float *out) {
  for (int id = 0; id < in.size(); id++) {
    out[id] = ach::compute(id, in);
  }
}

void simulate(ach::temperature_grid_f temp_in, float *temp_out,
              cudaStream_t stream) {
  single_thread_kernel<<<1, 1, 0, stream>>>(temp_in, temp_out);
}
