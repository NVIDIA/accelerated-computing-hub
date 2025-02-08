#include "ach.h"

void simulate(int height, int width,
              const thrust::universal_vector<float> &in,
                    thrust::universal_vector<float> &out)
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);

  thrust::tabulate(thrust::device, out.begin(), out.end(), [temp_in] __host__ __device__(int id) {
    int column = id % temp_in.extent(1);
    int row    = id / temp_in.extent(1);

    if (row > 0 && column > 0 && row < temp_in.extent(0) - 1 && column < temp_in.extent(1) - 1)
    {
      float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
      float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);

      return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
    }
    else
    {
      return temp_in(row, column);
    }
  });
}
