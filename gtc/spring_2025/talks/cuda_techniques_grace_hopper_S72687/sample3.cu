#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void kernel(int *data1, int *data2) {
  data1[threadIdx.x] = threadIdx.x;
  data2[threadIdx.x] = threadIdx.x;
}

int main() {
  int device_id;
  cudaGetDevice(&device_id);

  int *data = (int *)malloc(128 * sizeof(int));
  cudaMemLocation loc;
  loc.type = cudaMemLocationTypeHost;
  cudaMemAdvise_v2(data, 128 * sizeof(int), cudaMemAdviseSetPreferredLocation,
                   loc);
  cudaMemPrefetchAsync_v2(data, 128 * sizeof(int), loc, 0);

  int data2[128];
  cudaMemLocation loc2;
  loc2.id = device_id;
  loc2.type = cudaMemLocationTypeDevice;
  cudaMemAdvise_v2(data2, 128 * sizeof(int), cudaMemAdviseSetPreferredLocation,
                   loc2);
  cudaMemPrefetchAsync_v2(data2, 128 * sizeof(int), loc2, 0);

  kernel<<<1, 128>>>(data, data2);
  cudaDeviceSynchronize();

  for (int i = 0; i < 128; i++) {
    printf("(%d, %d)  ", data[i], data2[i]);
  }
  printf("\n");

  free(data);
  return 0;
}
