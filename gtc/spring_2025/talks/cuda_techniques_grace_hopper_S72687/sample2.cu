#include <cstdio>
#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void kernel(int *data1, int *data2)
{
  data1[threadIdx.x] = threadIdx.x;
  data2[threadIdx.x] = threadIdx.x;
}

int main()
{
  int *data = (int *)malloc(128 * sizeof(int));

  int data2[128];
  kernel<<<1, 128>>>(data, data2);
  cudaDeviceSynchronize();

  for (int i = 0; i < 128; i++)
  {
    printf("(%d, %d)  ", data[i], data2[i]);
  }
  printf("\n");
  
  free(data);
  return 0;
}

