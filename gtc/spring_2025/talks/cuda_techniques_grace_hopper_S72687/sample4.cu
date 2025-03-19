#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime_api.h>

__global__ void kernel(int *data) {
  cuda::atomic_ref<int, cuda::thread_scope_system> d(*data);
  d.fetch_add(1);
}

void add(int *data, int value) {
  cuda::atomic_ref<int, cuda::thread_scope_system> d(*data);
  for (int i = 0; i < value; i++) {
    // Multiple adds here to test the coherency protocol
    d.fetch_add(1);
  }
}

int main() {
  int device_id;
  cudaGetDevice(&device_id);

  int isHostNativeAtomicSupported;
  cudaDeviceGetAttribute(&isHostNativeAtomicSupported,
                         cudaDevAttrHostNativeAtomicSupported, device_id);
  printf("cudaDevAttrHostNativeAtomicSupported: %d\n",
         isHostNativeAtomicSupported);

  int *data = (int *)malloc(sizeof(int));
  // cudaMemLocation loc;
  // loc.type = cudaMemLocationTypeHost;
  // cudaMemAdvise_v2(data, sizeof(int), cudaMemAdviseSetPreferredLocation,
  // loc); cudaMemPrefetchAsync_v2(data, sizeof(int), loc, 0);
  cudaMemLocation loc;
  loc.id = device_id;
  loc.type = cudaMemLocationTypeDevice;
  cudaMemAdvise_v2(data, sizeof(int), cudaMemAdviseSetPreferredLocation, loc);
  cudaMemPrefetchAsync_v2(data, sizeof(int), loc, 0);

  int niters = 100;
  int warmup = 2;
  int grid = 4096;
  int block = 128;

  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < niters; iter++) {
    if (iter == warmup)
      start = std::chrono::high_resolution_clock::now();

    // Initialize to 0
    cuda::atomic_ref<int, cuda::thread_scope_system> atomic_data(*data);
    atomic_data.store(0);

    // Launch kernel atomics
    kernel<<<grid, block>>>(data);
    // Run CPU atomics
    add(data, grid);
    cudaDeviceSynchronize();
  }
  auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> duration = end - start;

  std::cout << "Final value: " << *data << ", expected " << (block + 1) * grid
            << std::endl;
  std::cout << "Time: " << duration.count() * 1000. / (niters - warmup) << " milliseconds"
            << std::endl;

  free(data);
  return 0;
}
