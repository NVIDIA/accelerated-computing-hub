#include "ach.h"

struct mean_functor {
    int width;
    __host__ __device__ float operator()(float x) const {
        return x / width;
    }
};

thrust::universal_vector<float> row_temperatures(
    int height, int width,
    thrust::universal_vector<int>& row_ids,
    thrust::universal_vector<float>& temp)
{
    thrust::universal_vector<float> means(height);

    // use `transform_output_iterator` instead of `means.begin()`
    auto means_output = means.begin(); 

    auto row_ids_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), 
        [=]__host__ __device__(int i) {
            return i / width;
        });
    auto row_ids_end = row_ids_begin + temp.size();

    thrust::reduce_by_key(thrust::device, 
                          row_ids_begin, 
                          row_ids_end, 
                          temp.begin(), 
                          thrust::make_discard_iterator(), 
                          means_output);

    auto transform_op = mean_functor{width};

    // remove this `transform` call
    thrust::transform(thrust::device, 
                      means.begin(), 
                      means.end(), 
                      means.begin(), 
                      transform_op);

    return means;
}
