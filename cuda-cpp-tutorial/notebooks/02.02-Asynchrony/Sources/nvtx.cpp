#include "ach.h"

void simulate(int width, int height, const thrust::device_vector<float> &in,
              thrust::device_vector<float> &out) 
{
  cuda::std::mdspan temp_in(thrust::raw_pointer_cast(in.data()), height, width);
  cub::DeviceTransform::Transform(
      thrust::make_counting_iterator(0), out.begin(), width * height,
      [=] __host__ __device__(int id) { return ach::compute(id, temp_in); });
}

int main() 
{
  int height = 2048;
  int width = 8192;

  thrust::device_vector<float> d_prev = ach::init(height, width);
  thrust::device_vector<float> d_next(height * width);
  thrust::host_vector<float> h_prev(height * width);

  const int compute_steps = 750;
  const int write_steps = 3;
  for (int write_step = 0; write_step < write_steps; write_step++) 
  {
    nvtx3::scoped_range r{std::string("write step ") + std::to_string(write_step)};

    {
      // TODO: Annotate the "copy" step using nvtx range
      thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());
    }

    {
      // TODO: Annotate the "compute" step using nvtx range
      for (int compute_step = 0; compute_step < compute_steps; compute_step++) 
      {
        simulate(width, height, d_prev, d_next);
        d_prev.swap(d_next);
      }
    }

    {
      // TODO: Annotate the "write" step using nvtx range
      ach::store(write_step, height, width, h_prev);
    }

    {
      // TODO: Annotate the "wait" step using nvtx range
      cudaDeviceSynchronize();
    }
  }
}
