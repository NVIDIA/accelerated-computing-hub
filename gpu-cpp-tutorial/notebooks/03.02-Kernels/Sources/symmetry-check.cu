#include "ach.cuh"

// 1. convert the function below from a CPU function into a CUDA kernel
void symmetry_check_kernel(ach::temperature_grid_f temp, int row) {
  int column = 0;

  if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1) {
    printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
  }
}

void symmetry_check(ach::temperature_grid_f temp, cudaStream_t stream) {
  int target_row = 0;
  // 2. use triple chevron to launch the kernel
  symmetry_check_kernel(temp, target_row);
}
