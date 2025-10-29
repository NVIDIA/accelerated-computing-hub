#include "ach.h"

void store(int step, int height, int width, const thrust::host_vector<float> &data)
{
  std::ofstream file("/tmp/heat_" + std::to_string(step) + ".bin", std::ios::binary);
  file.write(reinterpret_cast<const char *>(data.data()), height * width * sizeof(float));
}

void simulate(int width,
              int height,
              const thrust::device_vector<float> &in,
                    thrust::device_vector<float> &out)
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  auto compute = [=] __host__ __device__(int id) {
    const int column = id % width;
    const int row    = id / width;

    if (row > 0 && column > 0 && row < height - 1 && column < width - 1)
    {
      float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
      float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);

      return temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
    }
    else
    {
      return temp_in(row, column);
    }
  };

  auto cell_ids = thrust::make_counting_iterator(0);
  cub::DeviceTransform::Transform(cell_ids, out.begin(), width * height, compute);
}

int main()
{
  int height = 2048;
  int width  = 8192;

  thrust::device_vector<float> dprev(height * width, 15.0f);
  thrust::fill_n(dprev.begin(), width, 90.0f);
  thrust::fill_n(dprev.begin() + width * (height - 1), width, 90.0f);
  thrust::device_vector<float> dnext(height * width);
  thrust::host_vector<float> hprev(height * width);

  const int compute_steps = 500;
  const int write_steps = 3;
  for (int write_step = 0; write_step < write_steps; write_step++)
  {
    auto step_begin = std::chrono::high_resolution_clock::now();
    thrust::copy(dprev.begin(), dprev.end(), hprev.begin());

    for (int compute_step = 0; compute_step < compute_steps; compute_step++)
    {
      simulate(width, height, dprev, dnext);
      dprev.swap(dnext);
    }

    auto write_begin = std::chrono::high_resolution_clock::now();
    store(write_step, height, width, hprev);
    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_seconds = std::chrono::duration<double>(write_end - write_begin).count();

    cudaDeviceSynchronize();
    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_seconds = std::chrono::duration<double>(step_end - step_begin).count();
    std::printf("compute + write %d in %g s\n", write_step, step_seconds);
    std::printf("          write %d in %g s\n", write_step, write_seconds);
  }
}
