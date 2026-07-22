// 1D shallow-water Rusanov step, solved entirely on the GPU with Thrust.
//
// fields live in CUDA managed memory, each step is a thrust::for_each over
// interior cells with a __device__ lambda for the Rusanov flux step


#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cmath>

__device__ inline void rusanov_face(double hL, double hR, double huL, double huR,
                                    double g, double& Fh, double& Fhu) {
    const double DRY  = 1e-6;
    const double hL_s = hL > DRY ? hL : DRY;
    const double hR_s = hR > DRY ? hR : DRY;
    const double uL = huL / hL_s, uR = huR / hR_s;
    const double cL = ::sqrt(g * hL_s), cR = ::sqrt(g * hR_s);
    const double a  = ::fmax(::fabs(uL) + cL, ::fabs(uR) + cR);
    Fh  = 0.5 * (huL + huR) - 0.5 * a * (hR - hL);
    Fhu = 0.5 * (huL * uL + 0.5 * g * hL * hL + huR * uR + 0.5 * g * hR * hR)
        - 0.5 * a * (huR - huL);
}

void gpu_swe_solve(const double* h0, const double* hu0,
                   double* h_out, double* hu_out,
                   long Np2, double dx, double dt, double g, long n_steps) {
    const long   N   = Np2 - 2;
    const double inv = dt / dx;

    double *h, *hu, *hn, *hun;
    cudaMallocManaged(&h,   Np2 * sizeof(double));
    cudaMallocManaged(&hu,  Np2 * sizeof(double));
    cudaMallocManaged(&hn,  Np2 * sizeof(double));
    cudaMallocManaged(&hun, Np2 * sizeof(double));
    for (long i = 0; i < Np2; ++i) { h[i] = h0[i]; hu[i] = hu0[i]; }

    for (long s = 0; s < n_steps; ++s) {
        double *H = h, *HU = hu, *HN = hn, *HUN = hun;

        // reflective boundary, on the device
        thrust::for_each(thrust::device, thrust::counting_iterator<long>(0),
            thrust::counting_iterator<long>(1), [=] __device__ (long) {
                H[0] = H[1];   H[Np2 - 1]  = H[Np2 - 2];
                HU[0] = -HU[1]; HU[Np2 - 1] = -HU[Np2 - 2];
                HN[0] = H[0];   HUN[0] = HU[0];
                HN[Np2 - 1] = H[Np2 - 1]; HUN[Np2 - 1] = HU[Np2 - 1];
            });

        // interior Rusanov step, on the device
        thrust::for_each(thrust::device, thrust::counting_iterator<long>(1),
            thrust::counting_iterator<long>(N + 1), [=] __device__ (long i) {
                double Fw_h, Fw_hu, Fe_h, Fe_hu;
                rusanov_face(H[i - 1], H[i],     HU[i - 1], HU[i],     g, Fw_h, Fw_hu);
                rusanov_face(H[i],     H[i + 1],  HU[i],     HU[i + 1], g, Fe_h, Fe_hu);
                HN[i]  = H[i]  - inv * (Fe_h  - Fw_h);
                HUN[i] = HU[i] - inv * (Fe_hu - Fw_hu);
            });

        double *t = h; h = hn; hn = t; t = hu; hu = hun; hun = t;
    }

    cudaDeviceSynchronize();
    for (long i = 0; i < Np2; ++i) { h_out[i] = h[i]; hu_out[i] = hu[i]; }
    // add cudaError checks
    cudaFree(h); cudaFree(hu); cudaFree(hn); cudaFree(hun);
}
