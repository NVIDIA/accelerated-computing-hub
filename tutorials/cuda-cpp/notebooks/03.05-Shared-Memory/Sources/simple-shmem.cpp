#include <cstdio>

__global__ void kernel()
{
  __shared__ int shared[4];
  shared[threadIdx.x] = threadIdx.x;
  __syncthreads();

  if (threadIdx.x == 0)
  {
    for (int i = 0; i < 4; i++) {
      std::printf("shared[%d] = %d\n", i, shared[i]);
    }
  }
}

int main() {
  kernel<<<1, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}
