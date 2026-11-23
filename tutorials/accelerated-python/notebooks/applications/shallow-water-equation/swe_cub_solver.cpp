// SPDX-License-Identifier: Apache-2.0
// 1D shallow-water Rusanov step, solved entirely on the GPU with CUB.
//
// Fields live in device memory for the lifetime of the solver: gpu_swe_init
// allocates and uploads, gpu_swe_steps integrates, gpu_swe_fetch copies the
// result back. Each step runs two cub::DeviceFor::Bulk passes: the first
// computes every face flux once, the second updates the cells from the
// stored fluxes.

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

// Device-resident state for one grid size.
struct SweCubState {
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

static SweCubState g_cub;

static void gpu_swe_release() {
    if (g_cub.h0)  CUDA_CHECK(cudaFree(g_cub.h0));
    if (g_cub.hu0) CUDA_CHECK(cudaFree(g_cub.hu0));
    if (g_cub.h)   CUDA_CHECK(cudaFree(g_cub.h));
    if (g_cub.hu)  CUDA_CHECK(cudaFree(g_cub.hu));
    if (g_cub.hn)  CUDA_CHECK(cudaFree(g_cub.hn));
    if (g_cub.hun) CUDA_CHECK(cudaFree(g_cub.hun));
    if (g_cub.Fh)  CUDA_CHECK(cudaFree(g_cub.Fh));
    if (g_cub.Fhu) CUDA_CHECK(cudaFree(g_cub.Fhu));
    g_cub = SweCubState{};
}

// Allocate for this grid size and upload the initial condition. A repeat call
// at the resident size does nothing.
void gpu_swe_init(const double* h0, const double* hu0, long Np2) {
    if (Np2 < 2) {
        std::fprintf(stderr, "Np2 < 2\n");
        std::abort();
    }
    // CUB sets the index type, ensure valid for 32-bit
    if (Np2 - 1 > INT_MAX) {
        std::fprintf(stderr, "N too large for 32-bit indexing\n");
        std::abort();
    }
    if (g_cub.Np2 == Np2) return;
    gpu_swe_release();

    const int    N      = static_cast<int>(Np2 - 2);
    const size_t bytes  = Np2 * sizeof(double);
    const size_t fbytes = (N + 1) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&g_cub.h0,  bytes));
    CUDA_CHECK(cudaMalloc(&g_cub.hu0, bytes));
    CUDA_CHECK(cudaMalloc(&g_cub.h,   bytes));
    CUDA_CHECK(cudaMalloc(&g_cub.hu,  bytes));
    CUDA_CHECK(cudaMalloc(&g_cub.hn,  bytes));
    CUDA_CHECK(cudaMalloc(&g_cub.hun, bytes));
    CUDA_CHECK(cudaMalloc(&g_cub.Fh,  fbytes));
    CUDA_CHECK(cudaMalloc(&g_cub.Fhu, fbytes));
    CUDA_CHECK(cudaMemcpy(g_cub.h0,  h0,  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_cub.hu0, hu0, bytes, cudaMemcpyHostToDevice));
    g_cub.Np2 = Np2;
}

// Copy the result of the last solve back to the host.
void gpu_swe_fetch(double* h_out, double* hu_out) {
    if (g_cub.Np2 == 0) {
        std::fprintf(stderr, "gpu_swe_fetch before gpu_swe_init\n");
        std::abort();
    }
    const size_t bytes = g_cub.Np2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h_out,  g_cub.h,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hu_out, g_cub.hu, bytes, cudaMemcpyDeviceToHost));
}

// Integrate n_steps from the initial condition. Everything stays on the device.
void gpu_swe_steps(double dx, double dt, double g, long n_steps) {
    if (g_cub.Np2 == 0) {
        std::fprintf(stderr, "gpu_swe_steps before gpu_swe_init\n");
        std::abort();
    }
    const int    N      = static_cast<int>(g_cub.Np2 - 2);
    const long   Np2    = g_cub.Np2;
    const double inv    = dt / dx;
    const size_t bytes  = Np2 * sizeof(double);

    // Restart from the initial condition so repeated timings do equal work.
    CUDA_CHECK(cudaMemcpy(g_cub.h,  g_cub.h0,  bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g_cub.hu, g_cub.hu0, bytes, cudaMemcpyDeviceToDevice));

    double *h = g_cub.h, *hu = g_cub.hu, *hn = g_cub.hn, *hun = g_cub.hun;
    double *Fh = g_cub.Fh, *Fhu = g_cub.Fhu;

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

    // Store the buffers in whichever order the swaps left them.
    g_cub.h = h; g_cub.hu = hu; g_cub.hn = hn; g_cub.hun = hun;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}
