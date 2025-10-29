#include "ach.h"

float variance(const thrust::universal_vector<float> &x, float mean) {
  auto squared_differences = thrust::make_transform_iterator(
      x.begin(), [mean] __host__ __device__(float value) {
        return (value - mean) * (value - mean);
      });

  return thrust::reduce(thrust::device, squared_differences,
                        squared_differences + x.size()) /
         x.size();
}

float mean(thrust::universal_vector<float> vec) {
  return thrust::reduce(thrust::device, vec.begin(), vec.end()) / vec.size();
}

int main() {
  float ambient_temp = 20;
  thrust::universal_vector<float> prev{42, 24, 50};
  thrust::universal_vector<float> next{0, 0, 0};

  std::printf("step  variance\n");
  for (int step = 0; step < 3; step++) {
    thrust::transform(thrust::device, prev.begin(), prev.end(), next.begin(),
                      [=] __host__ __device__(float temp) {
                        return temp + 0.5 * (ambient_temp - temp);
                      });
    std::printf("%d     %.2f\n", step, variance(next, mean(next)));
    next.swap(prev);
  }
}