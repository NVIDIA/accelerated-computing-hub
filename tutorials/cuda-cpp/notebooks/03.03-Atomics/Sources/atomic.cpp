#include <cuda/std/span>
#include <cuda/std/atomic>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void kernel(cuda::std::span<int> count)
{
    // Wrap data in atomic_ref
    cuda::std::atomic_ref<int> ref(count[0]);

    // Atomically increment the underlying value
    ref.fetch_add(1);
}

int main()
{
    thrust::device_vector<int> count(1);

    int threads_in_block = 256;
    int blocks_in_grid = 42;

    kernel<<<blocks_in_grid, threads_in_block>>>(
        cuda::std::span<int>{thrust::raw_pointer_cast(count.data()), 1});

    cudaDeviceSynchronize();

    thrust::host_vector<int> count_host = count;
    std::cout << "expected: " << threads_in_block * blocks_in_grid << std::endl;
    std::cout << "observed: " << count_host[0] << std::endl;
}
