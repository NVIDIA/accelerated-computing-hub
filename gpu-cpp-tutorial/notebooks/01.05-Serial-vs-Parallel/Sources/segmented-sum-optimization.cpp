#include "ach.h"

thrust::universal_vector<float> row_temperatures(
    int height, int width,
    thrust::universal_vector<int>& row_ids,
    thrust::universal_vector<float>& temp)
{
    thrust::universal_vector<float> sums(height);

    // Modify the line below to use counting and transform iterators to 
    // generates row indices `id / width` instead
    auto row_ids_begin = row_ids.begin(); 
    auto row_ids_end = row_ids_begin + temp.size();

    thrust::reduce_by_key(thrust::device, 
                          row_ids_begin, row_ids_end, 
                          temp.begin(), 
                          thrust::make_discard_iterator(), 
                          sums.begin());

    return sums;
}
