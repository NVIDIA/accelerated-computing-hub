// SPDX-License-Identifier: Apache-2.0
// Signed histogram of cell births on a binned periodic field, one thread per voxel: q holds the bin index of every
// voxel (int32, n^3, C order with k fastest) and hist holds K_BINS counters, zeroed by the caller. Each thread adds
// (-1)^dim at the birth bin of its eight cells into a shared histogram, which the block then flushes into hist.
// The body between the markers is the reference. Notebook 94 splices its own.

#include <cuda_runtime.h>

constexpr int K_BINS = 128;

__device__ __forceinline__ int wrap1(int i, int n) { return i + 1 == n ? 0 : i + 1; }

__global__ void ecc_hist_kernel(const int* __restrict__ q, int* __restrict__ hist, int n)
{
    __shared__ int sh[K_BINS];
    for (int b = threadIdx.x; b < K_BINS; b += blockDim.x) sh[b] = 0;
    __syncthreads();
    const long long t  = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long nn = static_cast<long long>(n) * n;
    if (t < nn * n) {
        const int k = static_cast<int>(t % n), j = static_cast<int>((t / n) % n), i = static_cast<int>(t / nn);
        const int i1 = wrap1(i, n), j1 = wrap1(j, n), k1 = wrap1(k, n);
        // BEGIN BODY
        const int v000 = q[i * nn + j * n + k],   v100 = q[i1 * nn + j * n + k];
        const int v010 = q[i * nn + j1 * n + k],  v110 = q[i1 * nn + j1 * n + k];
        const int v001 = q[i * nn + j * n + k1],  v101 = q[i1 * nn + j * n + k1];
        const int v011 = q[i * nn + j1 * n + k1], v111 = q[i1 * nn + j1 * n + k1];
        const int ex = min(v000, v100), ey = min(v000, v010), ez = min(v000, v001);
        const int fxy = min(ex, min(v010, v110));
        const int fxz = min(ex, min(v001, v101));
        const int fyz = min(ey, min(v001, v011));
        const int c   = min(fxy, min(min(v001, v101), min(v011, v111)));
        atomicAdd(&sh[v000], 1);
        atomicSub(&sh[ex], 1);
        atomicSub(&sh[ey], 1);
        atomicSub(&sh[ez], 1);
        atomicAdd(&sh[fxy], 1);
        atomicAdd(&sh[fxz], 1);
        atomicAdd(&sh[fyz], 1);
        atomicSub(&sh[c], 1);
        // END BODY
    }
    __syncthreads();
    for (int b = threadIdx.x; b < K_BINS; b += blockDim.x)
        if (sh[b] != 0) atomicAdd(&hist[b], sh[b]);
}
