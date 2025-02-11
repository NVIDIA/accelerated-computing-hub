#include "ach.h"

void simulate(ach::temperature_grid_f temp_in, float *temp_out,
              cudaStream_t stream) {
  auto cell_ids = thrust::make_counting_iterator(0);
  cub::DeviceTransform::Transform(
      cell_ids, temp_out, temp_in.size(),
      [temp_in] __host__ __device__(int cell_id) {
        return ach::compute(cell_id, temp_in);
      },
      stream);
}
