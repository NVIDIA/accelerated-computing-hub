#include "ach.h"

int main() {
  thrust::universal_vector<int> vec{1, 2, 3};
  thrust::for_each(
      thrust::device, vec.begin(), vec.end(), [] __device__(int val) {
        std::printf("printing %d on %s\n", val, ach::execution_space());
      });
}
