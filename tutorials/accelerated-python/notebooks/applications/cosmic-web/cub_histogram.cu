// SPDX-License-Identifier: Apache-2.0
// CUB's device histogram behind a host function: n int32 samples in [0, k_bins) at samples_ptr counted into
// k_bins int32 counters at hist_ptr, the pointers and the stream passed as integers the way CuPy exposes them.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/device/device_histogram.cuh>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

void cub_histogram(unsigned long long samples_ptr, unsigned long long hist_ptr, long long n, int k_bins,
                   unsigned long long stream)
{
    // The scratch is sized by CUB on the first call and kept for the process, so later calls launch only the kernels.
    static void* scratch = nullptr;
    static size_t scratch_bytes = 0;
    const int* samples = reinterpret_cast<const int*>(samples_ptr);
    int* hist = reinterpret_cast<int*>(hist_ptr);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    size_t needed = 0;
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(nullptr, needed, samples, hist, k_bins + 1, 0, k_bins, n, s));
    if (needed > scratch_bytes) {
        if (scratch != nullptr) CUDA_CHECK(cudaFree(scratch));
        CUDA_CHECK(cudaMalloc(&scratch, needed));
        scratch_bytes = needed;
    }
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(scratch, scratch_bytes, samples, hist, k_bins + 1, 0, k_bins, n, s));
}
