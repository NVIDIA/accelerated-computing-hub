#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void kernel(int *data)
{
    data[threadIdx.x] += 2;
}
int main()
{
    int N = 128;
    int *data = (int *)malloc(N * sizeof(int));
    int *d_data;

    cudaMalloc((void **)&d_data, N * sizeof(int));

    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

    kernel<<<1, N>>>(d_data);

    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    free(data);
}

