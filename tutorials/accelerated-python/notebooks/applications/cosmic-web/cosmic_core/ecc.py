# SPDX-License-Identifier: Apache-2.0
"""The reference: the token histogram of cell births in CuPy, the curve, the integer gate, the Gaussian expectation."""

import numpy as np

from .config import K_BINS
from .device import release_pool
from .testfield import field, cache_path


def ecc_cells_cupy(q):
    """Births of the eight cells each voxel owns, on the device: (vertex, (ex, ey, ez), (fxy, fxz, fyz), cube)."""
    import cupy as cp
    ex = cp.minimum(q, cp.roll(q, -1, 0))
    ey = cp.minimum(q, cp.roll(q, -1, 1))
    ez = cp.minimum(q, cp.roll(q, -1, 2))
    fxy = cp.minimum(ex, cp.roll(ex, -1, 1))
    fxz = cp.minimum(ex, cp.roll(ex, -1, 2))
    fyz = cp.minimum(ey, cp.roll(ey, -1, 2))
    c = cp.minimum(fxy, cp.roll(fxy, -1, 2))
    return q, (ex, ey, ez), (fxy, fxz, fyz), c


def ecc_hist_cupy(q, k=K_BINS):
    """Signed histogram of cell births on the device, int64 of length k: +1 per vertex and face, -1 per edge or cube."""
    import cupy as cp
    v, (ex, ey, ez), (fxy, fxz, fyz), c = ecc_cells_cupy(q)

    def bc(a):
        return cp.bincount(a.ravel(), minlength=k)

    return bc(v) - bc(ex) - bc(ey) - bc(ez) + bc(fxy) + bc(fxz) + bc(fyz) - bc(c)


def ecc_from_hist(hist):
    """χ at every bin edge: the cells born at or above the edge, summed from the top."""
    h = np.asarray(hist, dtype=np.int64)
    return np.cumsum(h[::-1])[::-1]


def chi_reference(n):
    """The reference curve of the test field at size n, computed once and cached under data/."""
    import cupy as cp
    path = cache_path("chi", n, ".npy")
    if path.exists():
        return np.load(path)
    chi = ecc_from_hist(cp.asnumpy(ecc_hist_cupy(field(n))))
    release_pool()
    np.save(path, chi)
    return chi


def gate(n, hist):
    """(max |Δχ| against the reference at size n, Σ hist) of a tool's histogram: (0, 0) when the gate passes."""
    hist = np.asarray(hist, dtype=np.int64)
    diff = ecc_from_hist(hist) - chi_reference(n).astype(np.int64)
    return int(np.max(np.abs(diff))), int(hist.sum())


def euler_2d(mask):
    """(V, E, F, V - E + F) of a pixel mask: pixels, side-by-side pairs, 2x2 blocks."""
    m = np.asarray(mask, dtype=bool)
    v = int(m.sum())
    e = int((m[1:, :] & m[:-1, :]).sum() + (m[:, 1:] & m[:, :-1]).sum())
    f = int((m[1:, 1:] & m[:-1, 1:] & m[1:, :-1] & m[:-1, :-1]).sum())
    return v, e, f, v - e + f


def chi_gaussian(nu, lam, volume):
    """E[χ(ν)] of a Gaussian field: volume · λ³/(2π)² · (ν² − 1) · exp(−ν²/2)."""
    nu = np.asarray(nu, dtype=np.float64)
    return volume * lam ** 3 / (2.0 * np.pi) ** 2 * (nu ** 2 - 1.0) * np.exp(-0.5 * nu ** 2)


# Three times the rms relative difference of eight seeds at 256^3 against the expectation.
ANALYTIC_TOL = 0.105


def zero_crossings(edges, chi):
    """ν where the curve changes sign: interpolated between edges, or the edge itself where the value is zero."""
    chi = np.asarray(chi, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    sign = np.sign(chi)
    nonzero = np.nonzero(sign)[0]
    out = []
    for a, b in zip(nonzero[:-1], nonzero[1:]):
        if sign[a] * sign[b] >= 0:
            continue
        if b == a + 1:
            out.append(float(edges[a] + (edges[b] - edges[a]) * chi[a] / (chi[a] - chi[b])))
        else:
            out.append(float(edges[a + 1:b].mean()))
    return out
