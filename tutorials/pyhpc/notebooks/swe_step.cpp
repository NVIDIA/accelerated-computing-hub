// SPDX-License-Identifier: Apache-2.0
// 1D Shallow Water Equations — Rusanov flux, forward-Euler step.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <algorithm>

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
// The caller pre-allocates output buffers and re-applies BCs between
// steps. Ghost cells are carried through unchanged.
void cpp_step(
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> h_in,
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> hu_in,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> h_out,
    nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> hu_out,
    double dx, double dt, double g)
{
    if (h_in.ndim() != 1 || hu_in.ndim() != 1 ||
        h_out.ndim() != 1 || hu_out.ndim() != 1) {
        throw nb::value_error("cpp_step expects h, hu, h_new, and hu_new to be 1-D arrays");
    }

    const size_t Np2 = h_in.shape(0);
    if (hu_in.shape(0) != Np2 || h_out.shape(0) != Np2 ||
        hu_out.shape(0) != Np2) {
        throw nb::value_error("cpp_step expects h, hu, h_new, and hu_new to have the same length");
    }
    if (Np2 < 2) {
        throw nb::value_error("cpp_step arrays must contain at least two ghost cells");
    }

    const double* h  = h_in.data();
    const double* hu = hu_in.data();
    double* h_new  = h_out.data();
    double* hu_new = hu_out.data();
    const size_t N   = Np2 - 2;
    const double inv = dt / dx;

    h_new[0]      = h[0];      hu_new[0]      = hu[0];
    h_new[Np2-1]  = h[Np2-1];  hu_new[Np2-1]  = hu[Np2-1];

    for (size_t i = 1; i <= N; ++i) {
        double Fh_w, Fhu_w, Fh_e, Fhu_e;
        rusanov_face(h[i-1], h[i],   hu[i-1], hu[i],   g, Fh_w, Fhu_w);
        rusanov_face(h[i],   h[i+1], hu[i],   hu[i+1], g, Fh_e, Fhu_e);
        h_new[i]  = h[i]  - inv * (Fh_e  - Fh_w);
        hu_new[i] = hu[i] - inv * (Fhu_e - Fhu_w);
    }
}

NB_MODULE(swe_step, m) {
    m.doc() = "1D SWE Rusanov step (nanobind).";
    m.def("cpp_step", &cpp_step,
          nb::arg("h").noconvert(), nb::arg("hu").noconvert(),
          nb::arg("h_new").noconvert(), nb::arg("hu_new").noconvert(),
          nb::arg("dx"), nb::arg("dt"), nb::arg("g") = 9.81);
}
