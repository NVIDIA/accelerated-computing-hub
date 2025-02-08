#include "ach.h"

thrust::universal_vector<float> row_temperatures(
    int height, int width,
    thrust::universal_vector<int>& row_ids,
    thrust::universal_vector<float>& temp)
{
    thrust::universal_vector<float> sums(height);
    thrust::reduce_by_key(
        thrust::device, 
        row_ids.begin(), row_ids.end(),   // input keys 
        temp.begin(),                     // input values
        thrust::make_discard_iterator(),  // output keys
        sums.begin());                    // output values

    return sums;
}
