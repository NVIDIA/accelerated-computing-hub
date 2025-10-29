#include <cstdio>

__global__ void kernel(int value) {
  std::printf("value on device = %d\n", value);
}

int main() {
  int blocks_in_grid = 1;
  int threads_in_block = 1;
  cudaStream_t stream = 0;
  kernel<<<blocks_in_grid, threads_in_block, 0, stream>>>(42);
  cudaStreamSynchronize(stream);
}
