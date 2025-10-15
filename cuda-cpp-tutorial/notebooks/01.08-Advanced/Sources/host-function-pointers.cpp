#include <thrust/transform.h>
#include <thrust/universal_vector.h>
#include <cstdio>

__host__ __device__ float transformation(float x) {
  return 2 * x + 1;
}

int main() {
  thrust::universal_vector<float> vec{ 1, 2, 3 };

  thrust::transform(vec.begin(), vec.end(), vec.begin(), transformation);

  std::printf("%g %g %g\n", vec[0], vec[1], vec[2]);
}
