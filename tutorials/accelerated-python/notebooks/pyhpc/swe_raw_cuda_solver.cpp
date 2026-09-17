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

// Device-resident state for one grid size, as in swe_cub_solver.cpp.
struct SweRawState {
    double* h0  = nullptr;   // pristine initial condition
    double* hu0 = nullptr;
    double* h   = nullptr;   // working state
    double* hu  = nullptr;
    double* hn  = nullptr;   // second buffer
    double* hun = nullptr;
    double* Fh  = nullptr;   // face fluxes
    double* Fhu = nullptr;
    long    Np2 = 0;
};

static SweRawState g_raw;

static void gpu_swe_release_raw() {
    if (g_raw.h0)  CUDA_CHECK_RAW(cudaFree(g_raw.h0));
    if (g_raw.hu0) CUDA_CHECK_RAW(cudaFree(g_raw.hu0));
    if (g_raw.h)   CUDA_CHECK_RAW(cudaFree(g_raw.h));
    if (g_raw.hu)  CUDA_CHECK_RAW(cudaFree(g_raw.hu));
    if (g_raw.hn)  CUDA_CHECK_RAW(cudaFree(g_raw.hn));
    if (g_raw.hun) CUDA_CHECK_RAW(cudaFree(g_raw.hun));
    if (g_raw.Fh)  CUDA_CHECK_RAW(cudaFree(g_raw.Fh));
    if (g_raw.Fhu) CUDA_CHECK_RAW(cudaFree(g_raw.Fhu));
    g_raw = SweRawState{};
}

// Allocate for this grid size and upload the initial condition. A repeat call
// at the resident size does nothing.
void gpu_swe_init_raw(const double* h0, const double* hu0, long Np2) {
    if (Np2 < 2) {
        std::fprintf(stderr, "Np2 < 2\n");
        std::abort();
    }
    if (g_raw.Np2 == Np2) return;
    gpu_swe_release_raw();

    const long   N      = Np2 - 2;
    const size_t bytes  = Np2 * sizeof(double);
    const size_t fbytes = (N + 1) * sizeof(double);
    CUDA_CHECK_RAW(cudaMalloc(&g_raw.h0,  bytes));
    CUDA_CHECK_RAW(cudaMalloc(&g_raw.hu0, bytes));
    CUDA_CHECK_RAW(cudaMalloc(&g_raw.h,   bytes));
    CUDA_CHECK_RAW(cudaMalloc(&g_raw.hu,  bytes));
    CUDA_CHECK_RAW(cudaMalloc(&g_raw.hn,  bytes));
    CUDA_CHECK_RAW(cudaMalloc(&g_raw.hun, bytes));
    CUDA_CHECK_RAW(cudaMalloc(&g_raw.Fh,  fbytes));
    CUDA_CHECK_RAW(cudaMalloc(&g_raw.Fhu, fbytes));
    CUDA_CHECK_RAW(cudaMemcpy(g_raw.h0,  h0,  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_RAW(cudaMemcpy(g_raw.hu0, hu0, bytes, cudaMemcpyHostToDevice));
    g_raw.Np2 = Np2;
}

// Copy the result of the last solve back to the host.
void gpu_swe_fetch_raw(double* h_out, double* hu_out) {
    if (g_raw.Np2 == 0) {
        std::fprintf(stderr, "gpu_swe_fetch_raw before gpu_swe_init_raw\n");
        std::abort();
    }
    const size_t bytes = g_raw.Np2 * sizeof(double);
    CUDA_CHECK_RAW(cudaMemcpy(h_out,  g_raw.h,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RAW(cudaMemcpy(hu_out, g_raw.hu, bytes, cudaMemcpyDeviceToHost));
}

// Integrate n_steps from the initial condition. Everything stays on the device.
void gpu_swe_steps_raw(double dx, double dt, double g, long n_steps) {
    if (g_raw.Np2 == 0) {
        std::fprintf(stderr, "gpu_swe_steps_raw before gpu_swe_init_raw\n");
        std::abort();
    }
    const long   Np2    = g_raw.Np2;
    const long   N      = Np2 - 2;
    const double inv    = dt / dx;
    const size_t bytes  = Np2 * sizeof(double);

    // Restart from the initial condition so repeated timings do equal work.
    CUDA_CHECK_RAW(cudaMemcpy(g_raw.h,  g_raw.h0,  bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK_RAW(cudaMemcpy(g_raw.hu, g_raw.hu0, bytes, cudaMemcpyDeviceToDevice));

    double *h = g_raw.h, *hu = g_raw.hu, *hn = g_raw.hn, *hun = g_raw.hun;
    double *Fh = g_raw.Fh, *Fhu = g_raw.Fhu;

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

    // Store the buffers in whichever order the swaps left them.
    g_raw.h = h; g_raw.hu = hu; g_raw.hn = hn; g_raw.hun = hun;
    CUDA_CHECK_RAW(cudaDeviceSynchronize());
    CUDA_CHECK_RAW(cudaGetLastError());
}
