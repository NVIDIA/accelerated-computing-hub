// SPDX-License-Identifier: Apache-2.0
// 1D Shallow Water Equation solver with CUDA kernels: identical two-pass
// algorithm to swe_cub_solver.cpp without CUB abstractions.
// Use __global__ kernels with <<<grid, block>>>() launch syntax.
// The trade-off is that you lose the occupancy-driven launch sizing,
// the algorithm library, and friendly abstractions.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK_RAW(call)                                                 \
    do {                                                                     \
        const cudaError_t err_ = (call);                                     \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",               \
                         cudaGetErrorString(err_), __FILE__, __LINE__);      \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

__device__ inline void rusanov_face_raw(double hL, double hR, double huL, double huR,
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

// Pass 1: one flux per face f = 0..N, reflective ghost values derived on read.
__global__ void swe_faces_kernel(const double* __restrict__ H, const double* __restrict__ HU,
                                 double* __restrict__ FH, double* __restrict__ FHU,
                                 long N, double g) {
    const long f = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (f > N) return;
    const double hL  = (f == 0) ?  H[1]  : H[f];
    const double huL = (f == 0) ? -HU[1] : HU[f];
    const double hR  = (f == N) ?  H[N]  : H[f + 1];
    const double huR = (f == N) ? -HU[N] : HU[f + 1];
    rusanov_face_raw(hL, hR, huL, huR, g, FH[f], FHU[f]);
}

// Pass 2: update the interior from the stored fluxes; carry ghosts.
__global__ void swe_update_kernel(const double* __restrict__ H, const double* __restrict__ HU,
                                  const double* __restrict__ FH, const double* __restrict__ FHU,
                                  double* __restrict__ HN, double* __restrict__ HUN,
                                  long N, long Np2, double inv) {
    const long i = 1 + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N) return;
    HN[i]  = H[i]  - inv * (FH[i]  - FH[i - 1]);
    HUN[i] = HU[i] - inv * (FHU[i] - FHU[i - 1]);
    if (i == 1) { HN[0] = H[1]; HUN[0] = -HU[1]; }
    if (i == N) { HN[Np2 - 1] = H[N]; HUN[Np2 - 1] = -HU[N]; }
}

void gpu_swe_solve_raw(const double* h0, const double* hu0,
                       double* h_out, double* hu_out,
                       long Np2, double dx, double dt, double g, long n_steps) {
    if (Np2 < 2) {
        std::fprintf(stderr, "Np2 < 2\n");
        std::abort();
    }
    const long   N      = Np2 - 2;
    const double inv    = dt / dx;
    const size_t bytes  = Np2 * sizeof(double);
    const size_t fbytes = (N + 1) * sizeof(double);

    double *h, *hu, *hn, *hun, *Fh, *Fhu;
    CUDA_CHECK_RAW(cudaMalloc(&h,   bytes));
    CUDA_CHECK_RAW(cudaMalloc(&hu,  bytes));
    CUDA_CHECK_RAW(cudaMalloc(&hn,  bytes));
    CUDA_CHECK_RAW(cudaMalloc(&hun, bytes));
    CUDA_CHECK_RAW(cudaMalloc(&Fh,  fbytes));
    CUDA_CHECK_RAW(cudaMalloc(&Fhu, fbytes));
    CUDA_CHECK_RAW(cudaMemcpy(h,  h0,  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RAW(cudaMemcpy(hu, hu0, bytes, cudaMemcpyHostToDevice));

    const int block   = 256;
    const int grid_c  = (int)((N + block - 1) / block);
    const int grid_f  = (int)((N + block) / block);
    for (long s = 0; s < n_steps; ++s) {
        swe_faces_kernel<<<grid_f, block>>>(h, hu, Fh, Fhu, N, g);
        CUDA_CHECK_RAW(cudaGetLastError());
        swe_update_kernel<<<grid_c, block>>>(h, hu, Fh, Fhu, hn, hun, N, Np2, inv);
        CUDA_CHECK_RAW(cudaGetLastError());
        double *t = h; h = hn; hn = t; t = hu; hu = hun; hun = t;
    }

    CUDA_CHECK_RAW(cudaMemcpy(h_out,  h,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RAW(cudaMemcpy(hu_out, hu, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RAW(cudaDeviceSynchronize());
    CUDA_CHECK_RAW(cudaGetLastError());
    CUDA_CHECK_RAW(cudaFree(h));  CUDA_CHECK_RAW(cudaFree(hu));
    CUDA_CHECK_RAW(cudaFree(hn)); CUDA_CHECK_RAW(cudaFree(hun));
    CUDA_CHECK_RAW(cudaFree(Fh)); CUDA_CHECK_RAW(cudaFree(Fhu));
}
