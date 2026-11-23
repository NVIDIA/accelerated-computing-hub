# SPDX-License-Identifier: Apache-2.0
"""The fields on the device: the BBKS spectrum, the blur and standardisation every field goes through, the cloud-in-cell
deposit, the Gaussian test field of the sweeps, binning and the caches.

`field(n)` is the binned field every tool runs on: the Quijote cube at the benchmark size, the Gaussian test field otherwise."""

import numpy as np

from .config import K_BINS, NU_LO, NU_HI, R_SMOOTH_VOXELS, SEED, BOX_MPC, N_S, GAMMA, DATA_DIR, CANONICAL_N
from .device import release_pool

CHUNK = 1 << 24   # particles per deposit pass: the device holds one chunk of positions and its indices at a time


def _xp(a):
    """numpy or cupy, whichever module owns `a`."""
    import cupy
    return cupy.get_array_module(a)


# --- The spectrum -----------------------------------------------------------------

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


# --- Blur, standardisation and deposit on the device --------------------------------

def _kvectors_device(n):
    """`kvectors(n)` as float32 CuPy arrays."""
    import cupy as cp
    return tuple(cp.asarray(k, dtype=cp.float32) for k in kvectors(n))


def _blur_hat(fhat, r_voxels):
    """(blurred field, σ₀², σ₁²) from a field's rfftn spectrum `fhat`, which is blurred in place.

    The blur exp(−k²r²/2) is applied slab by slab and the variances are summed from the blurred spectrum by Parseval, so
    the spectrum is the only full-size array."""
    import cupy as cp
    n = fhat.shape[0]
    kx, ky, kz = _kvectors_device(n)
    weights = cp.asarray(rfft_weights(n), dtype=cp.float32)[None, None, :]
    half_r2 = np.float32(0.5 * r_voxels ** 2)
    sigma0_sq = sigma1_sq = 0.0
    step = max(1, n // 16)
    for i0 in range(0, n, step):
        sl = slice(i0, i0 + step)
        k2 = kx[sl] ** 2 + ky ** 2 + kz ** 2
        fhat[sl] *= cp.exp(-half_r2 * k2)
        power = cp.abs(fhat[sl])
        power *= power
        power *= weights
        sigma0_sq += float(power.sum(dtype=cp.float64))
        power *= k2
        sigma1_sq += float(power.sum(dtype=cp.float64))
        del power, k2
    return cp.fft.irfftn(fhat, s=(n, n, n)), sigma0_sq / n ** 6, sigma1_sq / n ** 6


def blur_device(delta, r_voxels=R_SMOOTH_VOXELS):
    """(blurred field, σ₀² = ⟨f²⟩, σ₁² = ⟨|∇f|²⟩) of a device field, blurred over r_voxels in k-space."""
    import cupy as cp
    return _blur_hat(cp.fft.rfftn(delta), r_voxels)


def standardise_device(field, sigma0_sq, sigma1_sq):
    """ν = (f − ⟨f⟩)/σ₀, standardised in place, and λ."""
    import cupy as cp
    field -= field.mean(dtype=cp.float64).astype(field.dtype)
    field *= field.dtype.type(1.0 / np.sqrt(sigma0_sq))
    return field, lam_from_sigmas(sigma0_sq, sigma1_sq)


def cic_density_device(positions, n, box, chunk=CHUNK):
    """Cloud-in-cell density of unit-mass particles on a periodic n³ grid: CuPy float32, mean one.

    `positions` is (N, 3) in the units of `box`, the side of the periodic box. The positions are deposited `chunk` at a time
    with `cupyx.scatter_add`, so the device holds one chunk of positions and indices."""
    import cupy as cp
    import cupyx
    rho = cp.zeros(n ** 3, dtype=cp.float32)
    scale = np.float32(n / box)
    for start in range(0, positions.shape[0], chunk):
        x = cp.asarray(positions[start:start + chunk]) * scale     # voxel units
        lo = cp.floor(x)
        frac = x - lo
        lo = lo.astype(cp.int32) % n
        hi = (lo + 1) % n
        for corner in range(8):
            w = cp.ones(x.shape[0], dtype=cp.float32)
            idx = cp.zeros(x.shape[0], dtype=cp.int64)
            for axis in range(3):
                upper = (corner >> axis) & 1
                w *= frac[:, axis] if upper else 1.0 - frac[:, axis]
                idx = idx * n + (hi[:, axis] if upper else lo[:, axis])
            cupyx.scatter_add(rho, idx, w)
        del x, lo, frac, hi, w, idx
    rho *= np.float32(n ** 3 / positions.shape[0])
    release_pool()
    return rho.reshape(n, n, n)


def nu_from_positions(positions, n, box):
    """(ν, λ), ν float32 on the host: the positions deposited on an n³ grid, the contrast blurred over R_SMOOTH_VOXELS
    and standardised, all on the device."""
    import cupy as cp
    delta = cic_density_device(positions, n, box)
    delta -= np.float32(1.0)
    nu, lam = standardise_device(*blur_device(delta))
    del delta
    nu = cp.asnumpy(nu)
    release_pool()
    return nu, lam


# --- The Gaussian test field ------------------------------------------------------------

def make_nu_field_device(n):
    """ν (CuPy float32, n³) and λ of the Gaussian test field: white noise shaped by the BBKS spectrum, then blurred and
    standardised, on the device."""
    import cupy as cp
    rng = cp.random.default_rng(SEED)
    fhat = cp.fft.rfftn(rng.standard_normal((n, n, n), dtype=cp.float32))
    kx, ky, kz = _kvectors_device(n)
    scale = np.float32(n / BOX_MPC)
    step = max(1, n // 16)   # slab by slab, so that the spectrum is the only full-size array
    for i0 in range(0, n, step):
        sl = slice(i0, i0 + step)
        k = cp.sqrt(kx[sl] ** 2 + ky ** 2 + kz ** 2) * scale
        fhat[sl] *= cp.sqrt(power_spectrum(k)).astype(cp.float32)
    return standardise_device(*_blur_hat(fhat, R_SMOOTH_VOXELS))


# --- Binning and caches ----------------------------------------------------------------

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


def field_cache_path(kind, n, suffix):
    """The cache file of `kind` for `field(n)`: the Quijote files at CANONICAL_N are named apart from the Gaussian ones."""
    if n != CANONICAL_N:
        return cache_path(kind, n, suffix)
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / f"quijote_{kind}_n{n}_k{K_BINS}_r{R_SMOOTH_VOXELS:g}{suffix}"


def binned_field(path, make_nu):
    """(q, λ) from the cache file `path`, else binned from make_nu() → (ν on the device, λ) and saved there as uint8.

    `q` is int32 on the device."""
    import cupy as cp
    if path.exists():
        with np.load(path) as f:
            q, lam = cp.asarray(f["q"]).astype(cp.int32), float(f["lam"])
    else:
        nu, lam = make_nu()
        q = quantise(nu)
        del nu
        np.savez(path, q=cp.asnumpy(q.astype(cp.uint8)), lam=lam)
    release_pool()
    return q, lam


def make_binned_field(n):
    """(q, λ): the binned Gaussian test field at size n, generated once per size and cached under data/."""
    return binned_field(cache_path("field", n, ".npz"), lambda: make_nu_field_device(n))


_FIELDS = {}   # one device copy of each binned field per size, shared by every runner in this process


def field(n):
    """The binned field at size n, int32 on the device: the Quijote cube at CANONICAL_N, the Gaussian test field otherwise."""
    if n not in _FIELDS:
        from .quijote import quijote_binned
        _FIELDS[n] = (quijote_binned() if n == CANONICAL_N else make_binned_field(n))[0]
    return _FIELDS[n]
