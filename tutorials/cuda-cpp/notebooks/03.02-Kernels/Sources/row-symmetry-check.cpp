#include "ach.cuh"

__global__ void symmetry_check_kernel(ach::temperature_grid_f temp, int row)
{
  // TODO: change the line below so that each thread in a grid checks exactly
  // one column
  int column = 0;

  if (abs(temp(row, column) - temp(temp.extent(0) - 1 - row, column)) > 0.1)
  {
    printf("Error: asymmetry in %d / %d\n", column, temp.extent(1));
  }
}

void symmetry_check(ach::temperature_grid_f temp, cudaStream_t stream)
{
  int width      = temp.extent(1);
  // TODO: launch sufficient number of threads to assign one thread per column

  int target_row = 0;
  symmetry_check_kernel<<<1, 1, 0, stream>>>(temp, target_row);
}
