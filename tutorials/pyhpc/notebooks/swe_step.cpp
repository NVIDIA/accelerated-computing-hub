// SPDX-License-Identifier: Apache-2.0
// Nanobind module for a 1D Shallow Water Equation solver in OpenMP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace nb = nanobind;

inline void rusanov_face(double hL, double hR,
                         double huL, double huR,
                         double g,
                         double& Fh, double& Fhu) {
    constexpr double DRY = 1e-6;
    const double hL_s = hL > DRY ? hL : DRY;
    const double hR_s = hR > DRY ? hR : DRY;
    const double uL = huL / hL_s;
    const double uR = huR / hR_s;
    const double cL = std::sqrt(g * hL_s);
    const double cR = std::sqrt(g * hR_s);
    const double a  = std::max(std::abs(uL) + cL, std::abs(uR) + cR);
    Fh  = 0.5 * (huL + huR) - 0.5 * a * (hR - hL);
    Fhu = 0.5 * (huL * uL + 0.5 * g * hL * hL
              +  huR * uR + 0.5 * g * hR * hR) - 0.5 * a * (huR - huL);
}

// One forward-Euler Rusanov step on 1D arrays of shape (N+2,):
//     [ghost, h[1], h[2], ..., h[N], ghost]
// Each interface flux is computed exactly once, as in swe_core.step_numpy:
// pass 1 writes the flux into caller-provided face buffers (length >= N+1),
// pass 2 differences the stored fluxes. The caller pre-allocates all
// buffers and re-applies BCs between steps; ghost cells are carried
// through unchanged.
void cpp_step(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig> h_in,
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig> hu_in,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> h_out,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> hu_out,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> Fh_buf,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> Fhu_buf,
    double dx, double dt, double g)
{
    const double* h  = h_in.data();
    const double* hu = hu_in.data();
    double* h_new  = h_out.data();
    double* hu_new = hu_out.data();
    double* Fh  = Fh_buf.data();
    double* Fhu = Fhu_buf.data();
    const size_t Np2 = h_in.shape(0);
    const size_t N   = Np2 - 2;
    const double inv = dt / dx;

    if (Np2 < 2)
        throw std::invalid_argument("state arrays need at least 2 cells");
    if (hu_in.shape(0) != Np2 || h_out.shape(0) != Np2 || hu_out.shape(0) != Np2)
        throw std::invalid_argument("state arrays must share one length");
    if (Fh_buf.shape(0) < N + 1 || Fhu_buf.shape(0) < N + 1)
        throw std::invalid_argument("face buffers need at least N+1 elements");

    // Pass 1: one flux per interface i+1/2, i = 0..N.
    #pragma omp parallel for
    for (size_t f = 0; f <= N; ++f)
        rusanov_face(h[f], h[f + 1], hu[f], hu[f + 1], g, Fh[f], Fhu[f]);

    // Pass 2: difference the stored fluxes over the interior.
    h_new[0]      = h[0];      hu_new[0]      = hu[0];
    h_new[Np2-1]  = h[Np2-1];  hu_new[Np2-1]  = hu[Np2-1];
    #pragma omp parallel for
    for (size_t i = 1; i <= N; ++i) {
        h_new[i]  = h[i]  - inv * (Fh[i]  - Fh[i - 1]);
        hu_new[i] = hu[i] - inv * (Fhu[i] - Fhu[i - 1]);
    }
}

NB_MODULE(swe_step, m) {
    m.doc() = "1D SWE Rusanov step (nanobind).";
    m.def("cpp_step", &cpp_step,
          nb::arg("h"), nb::arg("hu"), nb::arg("h_new"), nb::arg("hu_new"),
          nb::arg("Fh"), nb::arg("Fhu"),
          nb::arg("dx"), nb::arg("dt"), nb::arg("g") = 9.81);
}
