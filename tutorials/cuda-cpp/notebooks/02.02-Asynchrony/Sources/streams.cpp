#include "ach.h"
#include <nvtx3/nvToolsExt.h>

void store(int step, int height, int width, const thrust::host_vector<float> &data)
{
  std::ofstream file("/tmp/heat_" + std::to_string(step) + ".bin", std::ios::binary);
  file.write(reinterpret_cast<const char *>(thrust::raw_pointer_cast(data.data())), height * width * sizeof(float));
}

void simulate(int width,
              int height,
              const thrust::device_vector<float> &in,
                    thrust::device_vector<float> &out,
              cudaStream_t stream)
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  auto out_ptr = thrust::raw_pointer_cast(out.data());
  auto compute = [=] __host__ __device__(int id) {
    const int column = id % width;
    const int row    = id / width;

    if (row > 0 && column > 0 && row < height - 1 && column < width - 1)
    {
      float d2tdx2 = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
      float d2tdy2 = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);

      out_ptr[id] = temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
    }
    else
    {
      out_ptr[id] = temp_in(row, column);
    }
  };
  cub::DeviceFor::Bulk(height * width, compute, stream);
}

int main()
{
  int height = 2048;
  int width  = 8192;

  cudaStream_t compute_stream;
  cudaStreamCreate(&compute_stream);

  cudaStream_t copy_stream;
  cudaStreamCreate(&copy_stream);

  thrust::device_vector<float> d_buffer(height * width);
  thrust::device_vector<float> d_prev(height * width, 15.0f);
  thrust::fill_n(d_prev.begin(), width, 90.0f);
  thrust::fill_n(d_prev.begin() + width * (height - 1), width, 90.0f);
  thrust::device_vector<float> d_next(height * width);
  thrust::host_vector<float> h_temp(height * width);

  const int compute_steps = 500;
  const int write_steps = 3;
  for (int write_step = 0; write_step < write_steps; write_step++)
  {
    auto step_begin = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer.data()), 
                    thrust::raw_pointer_cast(d_prev.data()), 
                    height * width * sizeof(float), 
                    cudaMemcpyDeviceToDevice,
                    compute_stream);
    cudaStreamSynchronize(compute_stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(h_temp.data()), 
                    thrust::raw_pointer_cast(d_buffer.data()), 
                    height * width * sizeof(float), 
                    cudaMemcpyDeviceToHost, 
                    copy_stream);

    for (int compute_step = 0; compute_step < compute_steps; compute_step++)
    {
      simulate(width, height, d_prev, d_next, compute_stream);
      d_prev.swap(d_next);
    }

    cudaStreamSynchronize(copy_stream);

    auto write_begin = std::chrono::high_resolution_clock::now();
    store(write_step, height, width, h_temp);
    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_seconds = std::chrono::duration<double>(write_end - write_begin).count();

    cudaStreamSynchronize(compute_stream);
    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_seconds = std::chrono::duration<double>(step_end - step_begin).count();
    std::printf("compute + write %d in %g s\n", write_step, step_seconds);
    std::printf("          write %d in %g s\n", write_step, write_seconds);
  }

  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(copy_stream);
}
