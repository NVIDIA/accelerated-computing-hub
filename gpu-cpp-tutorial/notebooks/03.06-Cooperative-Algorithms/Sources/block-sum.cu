#include <cub/block/block_reduce.cuh>

constexpr int block_threads = 128;

__global__ void block_sum()
{
  using block_reduce_t = cub::BlockReduce<int, block_threads>;
  using storage_t = block_reduce_t::TempStorage;
  
  __shared__ storage_t storage;

  int block_sum = block_reduce_t(storage).Sum(threadIdx.x);

  if (threadIdx.x == 0)
  {
    printf("block sum = %d\n", block_sum);
  }
}

int main() {
  block_sum<<<1, block_threads>>>();
  cudaDeviceSynchronize();
  return 0;
}
