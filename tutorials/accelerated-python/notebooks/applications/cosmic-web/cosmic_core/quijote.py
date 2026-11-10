# SPDX-License-Identifier: Apache-2.0
"""The Quijote cube: the particle positions of one N-body simulation, deposited on the device into the density field that
notebooks 90 to 95 analyse and benchmark."""

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

from .config import R_SMOOTH_VOXELS, CANONICAL_N, BOX_MPC, DATA_DIR
from .fields import nu_from_positions, binned_field, field_cache_path

# Realisation 0 of the fiducial cosmology at z = 0 (snapshot 004): the CDM positions in Mpc/h as float32, (512^3, 3). The
# file holds the snapshot's PartType1/Coordinates, read from its HDF5 parts with h5py and hdf5plugin, converted from
# kpc/h to Mpc/h and saved with np.save. Quijote nulls the low 7 bits of each position, so this is the data's own precision.
QUIJOTE_FILE = "quijote_fiducial_0_z0_positions.npy"
QUIJOTE_URL = f"https://github.com/aaronj0/accelerated-computing-hub/releases/download/sc26-data/{QUIJOTE_FILE}"
QUIJOTE_SHA256 = "dea0cc8ac6be0c6758da305e40f34ec334bf6ad7ef9e6caa56c108a9664a28e7"
QUIJOTE_CITATION = "Villaescusa-Navarro et al. 2020, ApJS 250, 2, doi:10.3847/1538-4365/ab9d82"
QUIJOTE_LABEL = (f"Quijote fiducial cosmology, realisation 0, z = 0: {CANONICAL_N}^3 particles in a {BOX_MPC:.0f} Mpc/h box, "
                 f"CIC on a {CANONICAL_N}^3 grid, R = {R_SMOOTH_VOXELS:g} voxels ({QUIJOTE_CITATION})")


def _download(url, path, sha256):
    """Download `url` to `path` with a progress line and verify its SHA-256; a wrong digest removes the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")

    def progress(blocks, block_size, total):
        done = f"{blocks * block_size / 2**30:.2f}"
        print(f"\r  downloading {path.name}: {done} of {total / 2**30:.2f} GiB" if total > 0 else
              f"\r  downloading {path.name}: {done} GiB", end="", flush=True)

    urllib.request.urlretrieve(url, part, progress)
    print()
    digest = hashlib.sha256()
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        part.unlink()
        raise RuntimeError(f"{path.name} from {url} has SHA-256 {digest.hexdigest()}, expected {sha256}")
    part.replace(path)


def quijote_positions():
    """The particle positions, float32 (N, 3) in Mpc/h, memory-mapped from $COSMIC_WEB_DATA or data/; downloaded once."""
    for d in (os.environ.get("COSMIC_WEB_DATA"), DATA_DIR):
        if d and (Path(d) / QUIJOTE_FILE).is_file():
            return np.load(Path(d) / QUIJOTE_FILE, mmap_mode="r")
    print(f"no {QUIJOTE_FILE} under $COSMIC_WEB_DATA or data/: downloading it")
    _download(QUIJOTE_URL, DATA_DIR / QUIJOTE_FILE, QUIJOTE_SHA256)
    return np.load(DATA_DIR / QUIJOTE_FILE, mmap_mode="r")


def _quijote_nu():
    """(ν, λ) of the Quijote cube on the CANONICAL_N grid, float32 on the host: generated on the device from the particle
    positions on first use and cached under data/."""
    path = field_cache_path("nu", CANONICAL_N, ".npz")
    if path.exists():
        with np.load(path) as f:
            return f["nu"], float(f["lam"])
    nu, lam = nu_from_positions(quijote_positions(), CANONICAL_N, BOX_MPC)
    np.savez(path, nu=nu, lam=lam)
    return nu, lam


def quijote_field():
    """(ν, λ, label): the Quijote density field, printed with its label; see `_quijote_nu`."""
    nu, lam = _quijote_nu()
    print(QUIJOTE_LABEL)
    return nu, lam, QUIJOTE_LABEL


def quijote_binned():
    """(q, λ): the Quijote cube binned, int32 on the device, cached under data/."""
    import cupy as cp

    def nu_device():
        nu, lam = _quijote_nu()
        return cp.asarray(nu), lam

    return binned_field(field_cache_path("field", CANONICAL_N, ".npz"), nu_device)
