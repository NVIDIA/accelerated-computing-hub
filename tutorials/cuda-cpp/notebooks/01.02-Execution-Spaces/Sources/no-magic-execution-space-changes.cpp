#include "ach.h"

int main() {
  // TODO: Replace ??? with CPU or GPU
  ach::where_am_I("???");

  thrust::universal_vector<int> vec{1};
  thrust::for_each(thrust::device, vec.begin(), vec.end(),
                   [] __host__ __device__(int) { ach::where_am_I("???"); });

  thrust::for_each(thrust::host, vec.begin(), vec.end(),
                   [] __host__ __device__(int) { ach::where_am_I("???"); });

  ach::where_am_I("???");
}
