// SPDX-License-Identifier: Apache-2.0
// 1D shallow-water Rusanov step, solved entirely on the GPU with CUB.
//
// Fields live in device memory, copied in once before the time
// loop and back after it. Each step runs two cub::DeviceFor::Bulk passes:
// the first computes every face flux once, the second updates the cells
// from the stored fluxes.

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/device/device_for.cuh>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t err_ = (call);                                     \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",               \
                         cudaGetErrorString(err_), __FILE__, __LINE__);      \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

__device__ inline void rusanov_face(double hL, double hR, double huL, double huR,
                                    double g, double& Fh, double& Fhu) {
    const double DRY  = 1e-6;
    const double hL_s = hL > DRY ? hL : DRY;
    const double hR_s = hR > DRY ? hR : DRY;
    const double uL = huL / hL_s, uR = huR / hR_s;
    const double cL = sqrt(g * hL_s), cR = sqrt(g * hR_s);
    const double a  = fmax(fabs(uL) + cL, fabs(uR) + cR);
    Fh  = 0.5 * (huL + huR) - 0.5 * a * (hR - hL);
    Fhu = 0.5 * (huL * uL + 0.5 * g * hL * hL + huR * uR + 0.5 * g * hR * hR)
        - 0.5 * a * (huR - huL);
}

void gpu_swe_solve(const double* h0, const double* hu0,
                   double* h_out, double* hu_out,
                   long Np2, double dx, double dt, double g, long n_steps) {
    if (Np2 < 2) {
        std::fprintf(stderr, "Np2 < 2\n");
        std::abort();
    }
    // CUB sets the index type, ensure valid for 32-bit
    if (Np2 - 1 > INT_MAX) {
        std::fprintf(stderr, "N too large for 32-bit indexing\n");
        std::abort();
    }
    const int    N      = static_cast<int>(Np2 - 2);
    const double inv    = dt / dx;
    const size_t bytes  = Np2 * sizeof(double);
    const size_t fbytes = (N + 1) * sizeof(double);

    double *h, *hu, *hn, *hun, *Fh, *Fhu;
    CUDA_CHECK(cudaMalloc(&h,   bytes));
    CUDA_CHECK(cudaMalloc(&hu,  bytes));
    CUDA_CHECK(cudaMalloc(&hn,  bytes));
    CUDA_CHECK(cudaMalloc(&hun, bytes));
    CUDA_CHECK(cudaMalloc(&Fh,  fbytes));
    CUDA_CHECK(cudaMalloc(&Fhu, fbytes));
    CUDA_CHECK(cudaMemcpy(h,  h0,  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hu, hu0, bytes, cudaMemcpyHostToDevice));

    for (long s = 0; s < n_steps; ++s) {
        double *H = h, *HU = hu, *HN = hn, *HUN = hun, *FH = Fh, *FHU = Fhu;

        // Pass 1: one flux per face f = 0..N (between cells f and f+1)
        CUDA_CHECK(cub::DeviceFor::Bulk(N + 1, [=] __device__ (int f) {
                const double hL  = (f == 0) ?  H[1]  : H[f];
                const double huL = (f == 0) ? -HU[1] : HU[f];
                const double hR  = (f == N) ?  H[N]  : H[f + 1];
                const double huR = (f == N) ? -HU[N] : HU[f + 1];
                rusanov_face(hL, hR, huL, huR, g, FH[f], FHU[f]);
            }));

        // Pass 2: update the interior from the stored fluxes
        CUDA_CHECK(cub::DeviceFor::Bulk(N, [=] __device__ (int j) {
                const int i = j + 1;
                HN[i]  = H[i]  - inv * (FH[i]  - FH[i - 1]);
                HUN[i] = HU[i] - inv * (FHU[i] - FHU[i - 1]);
                if (i == 1) { HN[0] = H[1]; HUN[0] = -HU[1]; }
                if (i == N) { HN[Np2 - 1] = H[N]; HUN[Np2 - 1] = -HU[N]; }
            }));

        double *t = h; h = hn; hn = t; t = hu; hu = hun; hun = t;
    }

    CUDA_CHECK(cudaMemcpy(h_out,  h,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hu_out, hu, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(h));  CUDA_CHECK(cudaFree(hu));
    CUDA_CHECK(cudaFree(hn)); CUDA_CHECK(cudaFree(hun));
    CUDA_CHECK(cudaFree(Fh)); CUDA_CHECK(cudaFree(Fhu));
}
