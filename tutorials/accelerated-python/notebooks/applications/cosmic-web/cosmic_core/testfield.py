# SPDX-License-Identifier: Apache-2.0
"""The Gaussian test field every tool is timed on: the BBKS spectrum, device-side generation and binning, the cache."""

import numpy as np

from .config import K_BINS, NU_LO, NU_HI, R_SMOOTH_VOXELS, SEED, BOX_MPC, N_S, GAMMA, DATA_DIR
from .device import release_pool


def _xp(a):
    """numpy or cupy, whichever module owns `a`."""
    import cupy
    return cupy.get_array_module(a)


def kvectors(n):
    """The three wavenumber components on the rfftn grid, broadcastable, radians per voxel."""
    k  = 2.0 * np.pi * np.fft.fftfreq(n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n)
    return k[:, None, None], k[None, :, None], kz[None, None, :]


def kgrid(n):
    """Wavenumber magnitude squared on the rfftn grid, in (radians per voxel)²."""
    kx, ky, kz = kvectors(n)
    return kx ** 2 + ky ** 2 + kz ** 2


def rfft_weights(n):
    """Multiplicity of each rfftn mode in the full spectrum: 2 for 0 < kz < Nyquist, else 1."""
    w = np.full(n // 2 + 1, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w


def bbks_transfer(q):
    """BBKS (1986, ApJ 304, 15, eq. G3) cold-dark-matter transfer function of q = k/Γ, on q's device."""
    xp = _xp(q)
    out = xp.ones_like(q)
    nz = q > 0
    x = 2.34 * q[nz]
    out[nz] = (xp.log1p(x) / x
               * (1.0 + 3.89 * q[nz] + (16.1 * q[nz]) ** 2
                  + (5.46 * q[nz]) ** 3 + (6.71 * q[nz]) ** 4) ** -0.25)
    return out


def power_spectrum(k):
    """Unnormalised linear power spectrum P(k) ∝ k^n_s T(k/Γ)², k in h/Mpc, on k's device."""
    xp = _xp(k)
    return xp.where(k > 0, k ** N_S * bbks_transfer(k / GAMMA) ** 2, 0.0)


def lam_from_sigmas(sigma0_sq, sigma1_sq):
    """λ = sqrt(σ₁² / 3σ₀²) in voxel⁻¹, the parameter of the Gaussian expectation."""
    return float(np.sqrt(sigma1_sq / (3.0 * sigma0_sq)))


def make_nu_field_device(n):
    """ν (CuPy float32, n³) and λ, generated on the device.

    White noise, the BBKS spectrum, a Gaussian smoothing, then standardised."""
    import cupy as cp
    rng = cp.random.default_rng(SEED)
    white = rng.standard_normal((n, n, n), dtype=cp.float32)
    fhat = cp.fft.rfftn(white)
    del white
    k = cp.asarray(2.0 * np.pi * np.fft.fftfreq(n), dtype=cp.float32)
    kz = cp.asarray(2.0 * np.pi * np.fft.rfftfreq(n), dtype=cp.float32)
    weights = cp.asarray(rfft_weights(n), dtype=cp.float32)[None, None, :]
    scale, half_r2 = np.float32(n / BOX_MPC), np.float32(0.5 * R_SMOOTH_VOXELS ** 2)
    sigma0_sq = sigma1_sq = 0.0
    step = max(1, n // 16)   # slab by slab, so that the spectrum is the only full-size array
    for i0 in range(0, n, step):
        sl = slice(i0, i0 + step)
        k2 = k[sl, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2
        amp = cp.sqrt(power_spectrum(cp.sqrt(k2) * scale)).astype(cp.float32)
        amp *= cp.exp(-half_r2 * k2)
        fhat[sl] *= amp
        del amp
        power = cp.abs(fhat[sl])
        power *= power
        power *= weights
        sigma0_sq += float(power.sum(dtype=cp.float64))
        power *= k2
        sigma1_sq += float(power.sum(dtype=cp.float64))
        del power, k2
    sigma0_sq /= n ** 6
    sigma1_sq /= n ** 6
    field = cp.fft.irfftn(fhat, s=(n, n, n))
    del fhat
    field -= field.mean(dtype=cp.float64).astype(cp.float32)
    field *= np.float32(1.0 / np.sqrt(sigma0_sq))
    return field, lam_from_sigmas(sigma0_sq, sigma1_sq)


def bin_edges(k=K_BINS):
    """Lower edge of every bin: bin b covers [edge_b, edge_b + (NU_HI − NU_LO)/k)."""
    return NU_LO + (NU_HI - NU_LO) * np.arange(k) / k


def quantise(nu, k=K_BINS):
    """Bin index of every voxel as int32, on whichever device holds ν."""
    xp = _xp(nu)
    # float64, so that the bin edges are exact
    idx = xp.floor((xp.asarray(nu, dtype=xp.float64) - NU_LO) * (k / (NU_HI - NU_LO)))
    return xp.clip(idx, 0, k - 1).astype(xp.int32)


def cache_path(kind, n, suffix):
    """data/<kind>_n<n>_k<K>_r<R>_seed<seed><suffix>, with data/ created."""
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / f"{kind}_n{n}_k{K_BINS}_r{R_SMOOTH_VOXELS:g}_seed{SEED}{suffix}"


# One device copy of the binned field per size, shared by every runner in this process. `bench.by_size` is not used
# here: it memoises the notebooks' own state and records its caller's line for the profiler probe.
_FIELDS = {}


def make_binned_field(n):
    """(q, lam): the binned test field at size n (int32, on the device) and its λ.

    Generated and binned on the device once per size, cached under data/ as uint8 and widened to int32 on load."""
    if n in _FIELDS:
        return _FIELDS[n]
    import cupy as cp
    path = cache_path("field", n, ".npz")
    if path.exists():
        with np.load(path) as f:
            q, lam = cp.asarray(f["q"]).astype(cp.int32), float(f["lam"])
    else:
        nu, lam = make_nu_field_device(n)
        q = quantise(nu)
        del nu
        np.savez(path, q=cp.asnumpy(q.astype(cp.uint8)), lam=lam)
    release_pool()
    _FIELDS[n] = (q, lam)
    return q, lam


def field(n):
    """The binned field at size n, int32 on the device, built once per size so that no timed call pays for it."""
    return make_binned_field(n)[0]
