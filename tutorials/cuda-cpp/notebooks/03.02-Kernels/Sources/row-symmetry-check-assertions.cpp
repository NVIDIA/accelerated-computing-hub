#define CCCL_ENABLE_ASSERTIONS
#include "ach.h"

__global__ void symmetry_check_kernel(ach::temperature_grid_f temp, int row) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1) {
    printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
  }
}

void symmetry_check(ach::temperature_grid_f temp_in, cudaStream_t stream) {
  int width = temp_in.extent(1);
  int block_size = 1024;
  int grid_size = cuda::ceil_div(width, block_size);

  int target_row = 0;
  symmetry_check_kernel<<<grid_size, block_size, 0, stream>>>(temp_in,
                                                              target_row);
}

void simulate(ach::temperature_grid_f temp_in, float *temp_out,
              cudaStream_t stream) {
  symmetry_check(temp_in, stream);
  ach::simulate(temp_in, temp_out, stream);
}
