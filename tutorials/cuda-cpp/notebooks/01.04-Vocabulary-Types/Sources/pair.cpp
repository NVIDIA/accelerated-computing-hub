#include "ach.h"

__host__ __device__
cuda::std::pair<int, int> row_col(int id, int width) {
    return cuda::std::make_pair(id / width, id % width);
}

void simulate(int height, int width,
              const thrust::universal_vector<float> &in,
                    thrust::universal_vector<float> &out)
{
  const float *in_ptr = thrust::raw_pointer_cast(in.data());

  thrust::tabulate(
    thrust::device, out.begin(), out.end(), 
    [in_ptr, height, width] __host__ __device__(int id) {
      auto [row, column] = row_col(id, width);

      if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
        float d2tdx2 = in_ptr[(row) * width + column - 1] - 2 * in_ptr[row * width + column] + in_ptr[(row) * width + column + 1];
        float d2tdy2 = in_ptr[(row - 1) * width + column] - 2 * in_ptr[row * width + column] + in_ptr[(row + 1) * width + column];

        return in_ptr[row * width + column] + 0.2f * (d2tdx2 + d2tdy2);
      } else {
        return in_ptr[row * width + column];
      }
    });
}
