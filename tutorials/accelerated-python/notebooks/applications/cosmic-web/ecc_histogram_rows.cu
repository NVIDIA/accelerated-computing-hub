// SPDX-License-Identifier: Apache-2.0
// The stencil and the shared histogram of ecc_histogram.cu on a 3D grid (gridDim.x, n, n): blockIdx.y and
// blockIdx.z carry j and i, so no thread divides to find its voxel, and each thread strides along k.
// The names shared with ecc_histogram.cu end in V: both files are compiled into the one interpreter, where a name is
// defined once.

#include <cuda_runtime.h>

constexpr int K_BINS_V = 128;

__device__ __forceinline__ int wrap1v(int i, int n) { return i + 1 == n ? 0 : i + 1; }

// The eight births of voxel (i, j, k) and their tokens into the block's shared histogram.
__device__ __forceinline__ void ecc_count_voxel(const int* __restrict__ q, int* sh, int i, int j, int k,
                                                int i1, int j1, int k1, int n, long long nn)
{
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
}

__global__ void ecc_hist_rows_kernel(const int* __restrict__ q, int* __restrict__ hist, int n)
{
    __shared__ int sh[K_BINS_V];
    for (int b = threadIdx.x; b < K_BINS_V; b += blockDim.x) sh[b] = 0;
    __syncthreads();
    const int j = blockIdx.y, i = blockIdx.z;
    const int i1 = wrap1v(i, n), j1 = wrap1v(j, n);
    const long long nn = static_cast<long long>(n) * n;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x)
        ecc_count_voxel(q, sh, i, j, k, i1, j1, wrap1v(k, n), n, nn);
    __syncthreads();
    for (int b = threadIdx.x; b < K_BINS_V; b += blockDim.x)
        if (sh[b] != 0) atomicAdd(&hist[b], sh[b]);
}
