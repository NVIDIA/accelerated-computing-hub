#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void kernel(int *data)
{
    data[threadIdx.x] += 2;
}
int main()
{
    int N = 128;

    int *data;
    cudaMallocManaged((void **)&data, N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        data[i] = i;
    }

    kernel<<<1, N>>>(data);

    cudaDeviceSynchronize();
    cudaFree(data);
}

