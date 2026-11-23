# SPDX-License-Identifier: Apache-2.0
"""The Zel'dovich stand-in of notebook 90: grid particles displaced along the linear flow, for the particle pictures and the
density field of its first sections."""

import numpy as np

from .config import R_SMOOTH_VOXELS, SEED, SIGMA8, BOX_MPC, DEMO_BOX_MPC
from .fields import kgrid, kvectors, rfft_weights, power_spectrum, nu_from_positions, cache_path


def gaussian_field(n, box_mpc=BOX_MPC):
    """Periodic Gaussian random field with the BBKS spectrum on the host: zero mean, unnormalised, float64."""
    from scipy import fft
    rng = np.random.default_rng(SEED)
    fhat = fft.rfftn(rng.standard_normal((n, n, n)), workers=-1)
    fhat *= np.sqrt(power_spectrum(np.sqrt(kgrid(n)) * (n / box_mpc)))
    return fft.irfftn(fhat, s=(n, n, n), workers=-1)


def _tophat_k(x):
    """Fourier transform of a unit top-hat sphere: 3 (sin x − x cos x) / x³."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nz = x > 1e-6
    out[nz] = 3.0 * (np.sin(x[nz]) - x[nz] * np.cos(x[nz])) / x[nz] ** 3
    return out


def zeldovich_displacement(n, box_mpc=BOX_MPC):
    """Linear Zel'dovich displacement field ψ (three n³ arrays, voxel units) for δ normalised to SIGMA8."""
    from scipy import fft
    delta_hat = fft.rfftn(gaussian_field(n, box_mpc), workers=-1)
    k2_phys = kgrid(n) * (n / box_mpc) ** 2
    w = rfft_weights(n)[None, None, :]
    var8 = float((w * (delta_hat.real ** 2 + delta_hat.imag ** 2)
                  * _tophat_k(np.sqrt(k2_phys) * 8.0) ** 2).sum()) / n ** 6
    delta_hat *= SIGMA8 / np.sqrt(var8)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_k2 = np.where(k2_phys > 0, 1.0 / k2_phys, 0.0)
    cell = box_mpc / n
    psi = []
    for k_axis in kvectors(n):
        k_phys = k_axis * (n / box_mpc)
        psi.append(fft.irfftn(1j * k_phys * inv_k2 * delta_hat, s=(n, n, n), workers=-1) / cell)
    return psi


def zeldovich_positions(n, growth=1.0, box_mpc=BOX_MPC):
    """(n³, 3) float32 positions in voxel units: the grid particles displaced by growth · ψ."""
    psi = zeldovich_displacement(n, box_mpc)
    grid = np.ogrid[:n, :n, :n]
    return np.stack([(g + growth * p).ravel() for g, p in zip(grid, psi)], axis=1).astype(np.float32)


def zeldovich_particles(n=128, start=0.0, end=1.0, interval=0.1):
    """[(D, (x, y))]: the particles that started on the middle plane, seen from above, one frame per growth factor D.

    D runs from start to end in steps of interval. The same particles are followed from frame to frame. Positions wrap
    around the periodic box."""
    psi = zeldovich_displacement(n, DEMO_BOX_MPC)
    plane = n // 2
    gx, gy = np.indices((n, n), dtype=np.float64)
    frames = []
    for d in np.round(np.linspace(start, end, int(round(abs(end - start) / interval)) + 1), 6):
        x = np.mod(gx + d * psi[0][:, :, plane], n)
        y = np.mod(gy + d * psi[1][:, :, plane], n)
        frames.append((float(d), (x.ravel().astype(np.float32), y.ravel().astype(np.float32))))
    return frames


def zeldovich_field(n=256):
    """(ν, λ, label): the Zel'dovich stand-in on an n³ grid, deposited and blurred on the device once and cached under
    data/."""
    path = cache_path("zeldovich", n, ".npz")
    if path.exists():
        with np.load(path) as f:
            nu, lam = f["nu"], float(f["lam"])
    else:
        nu, lam = nu_from_positions(zeldovich_positions(n), n, box=n)
        np.savez(path, nu=nu, lam=lam)
    label = (f"Zel'dovich stand-in: N = {n}^3, box {BOX_MPC:.0f} Mpc/h, sigma8 {SIGMA8}, "
             f"seed {SEED}, R = {R_SMOOTH_VOXELS:g} voxels")
    print(label)
    return nu, lam, label
