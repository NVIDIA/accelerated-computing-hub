#include "ach.h"

float median(thrust::universal_vector<float> vec) 
{
    // TODO: Make the below code execute on the GPU
    std::sort(vec.begin(), vec.end());
    return vec[vec.size() / 2];
}

int main() 
{
    float k = 0.5;
    float ambient_temp = 20;
    thrust::universal_vector<float> temp{ 42, 24, 50 };
    auto transformation = [=] __host__ __device__ (float temp) { return temp + k * (ambient_temp - temp); };

    std::printf("step  median\n");
    for (int step = 0; step < 3; step++) {
        thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), transformation);
        float median_temp = median(temp);
        std::printf("%d     %.2f\n", step, median_temp);
    }
}
