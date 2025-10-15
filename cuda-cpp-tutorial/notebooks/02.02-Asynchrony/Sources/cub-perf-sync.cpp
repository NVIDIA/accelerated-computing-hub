#include "ach.h"

float simulate(int width,
               int height,
               const thrust::device_vector<float> &in,
                     thrust::device_vector<float> &out,
               bool use_cub) 
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  auto compute = [=] __host__ __device__(int id) {
    const int column = id % width;
    const int row    = id / width;

    // loop over all points in domain (except boundary)
    if (row > 0 && column > 0 && row < height - 1 && column < width - 1)
    {
      // evaluate derivatives
      float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
      float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);

      // update temperatures
      return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
    }
    else
    {
      return temp_in(row, column);
    }
  };

  auto begin = std::chrono::high_resolution_clock::now();

  if (use_cub) 
  {
    auto cell_ids = thrust::make_counting_iterator(0);
    cub::DeviceTransform::Transform(cell_ids, out.begin(), width * height, compute);
    cudaDeviceSynchronize();
  }
  else 
  {
    thrust::tabulate(thrust::device, out.begin(), out.end(), compute);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<float>(end - begin).count();
}

int main()
{
  std::cout << "size, thrust, cub\n";
  for (int size = 1024; size <= 16384; size *= 2)
  {
    int width = size;
    int height = size;
    thrust::device_vector<float> current_temp(height * width, 15.0f);
    thrust::device_vector<float> next_temp(height * width);

    std::cout << size << ", "
              << simulate(width, height, current_temp, next_temp, false) << ", "
              << simulate(width, height, current_temp, next_temp, true) << "\n";
  }
}
