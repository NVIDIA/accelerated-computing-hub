"""1D Shallow Water Equations reference solver in NumPy.

We solve

    ∂h/∂t    + ∂(hu)/∂x                 = 0      (water height)
    ∂(hu)/∂t + ∂(hu²/h + ½ g h²)/∂x     = 0      (momentum)

on a domain [0, L] split into N cells of width dx = L/N, with one ghost
cell on each side. State arrays have shape (N+2,); the interior is [1:-1].

Time discretisation: forward Euler.
Space discretisation: finite volume with the Rusanov interface flux.

The discrete update at cell i is

    q_new[i] = q[i] - (dt/dx) (F[i+½] - F[i-½])

where q = (h, hu) is the conserved state. The Rusanov flux at face i+½
(between cells i and i+1) is

    F[i+½] = ½ (F(q_L) + F(q_R))  -  ½ a (q_R - q_L)
    a       = max(|u_L| + √(g h_L),  |u_R| + √(g h_R))

with u = hu/h the velocity.

`step_numpy` is the reference implementation. Each notebook in the
tutorial replaces it with a different tool and validates the result
against this one via `max_diff(h_tool, h_numpy) < TOL`.
"""

import json, time
from pathlib import Path
import numpy as np

g       = 9.81     # gravitational acceleration, m/s²
DRY_TOL = 1e-6     # depth below this is clamped (to avoid division by 0)

# As this is a canonical asset, allow arbitrary callers to resolve this file's location
HERE          = Path(__file__).resolve().parent
TIMINGS_PATH  = str(HERE / "timings.json")
SWE_STEP_CPP  = str(HERE / "swe_step.cpp")


# --- Initial / boundary conditions -----------------------------------------

def bump_ic(N, L=10.0, h0=1.0, amplitude=0.1, sigma=0.5):
    """1D Gaussian-bump pulse:
        h(x, 0) = h0 + amplitude * exp(-((x - L/2) / sigma)**2)
        hu(x, 0) = 0
    The bump splits into two counter-propagating wave packets at
    c = sqrt(g * h0). In the linear regime (amplitude << h0) they
    stay smooth indefinitely.
    """
    dx = L / N
    xs = (np.arange(N + 2) - 0.5) * dx
    h  = h0 + amplitude * np.exp(-((xs - L / 2) / sigma) ** 2)
    hu = np.zeros_like(h)
    return h, hu


def apply_bc_reflective(h, hu):
    """Reflective walls: mirrors ghost cells and flips momentum sign."""
    h[0]  =  h[1];   h[-1]  =  h[-2]
    hu[0] = -hu[1];  hu[-1] = -hu[-2]


# --- Time step (forward Euler + Rusanov flux) -------------------------------

def fixed_dt(h_max, dx, cfl=0.4, g=g):
    """One-shot dt from the max wave speed in the IC (CFL stability).
    h_max is the peak h value in the initial condition."""
    return float(cfl * dx / np.sqrt(g * h_max))


def step_numpy(h, hu, dx, dt, g=g, tol=DRY_TOL):
    """One forward-Euler step with Rusanov flux. Returns (h_new, hu_new)."""
    # Left/right states at every interface i+½  (length N+1).
    hL, hR   = h[:-1],  h[1:]
    huL, huR = hu[:-1], hu[1:]

    # Velocities and wave speeds at the face.
    h_safe_L = np.maximum(hL, tol)
    h_safe_R = np.maximum(hR, tol)
    uL, uR = huL / h_safe_L,        huR / h_safe_R
    cL, cR = np.sqrt(g * h_safe_L), np.sqrt(g * h_safe_R)
    a      = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

    # Rusanov interface flux: average of physical fluxes − stabilising diffusion.
    F_h  = 0.5 * (huL + huR)                                  - 0.5 * a * (hR  - hL)
    F_hu = 0.5 * (huL*uL + 0.5*g*hL*hL
                + huR*uR + 0.5*g*hR*hR)                       - 0.5 * a * (huR - huL)

    # Divergence: interior cells only; ghost cells unchanged.
    h_new, hu_new = h.copy(), hu.copy()
    h_new[1:-1]   = h[1:-1]  - (dt / dx) * (F_h[1:]  - F_h[:-1])
    hu_new[1:-1]  = hu[1:-1] - (dt / dx) * (F_hu[1:] - F_hu[:-1])
    return h_new, hu_new


# --- Validation -------------------------------------------------------------

def max_diff(a, b):
    """Max |a - b| as a float -  the cross-tool acceptance metric."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# --- Timing harness ---------------------------------------------------------

def timed_run(fn, *args, warmup=2, repeats=5, sync_fn=None, label=""):
    """Time fn(*args) with warmups + repeats. Returns median / min / max [s].

    GPU callers must pass a sync_fn so the timer waits for the kernel:
        JAX  : lambda: jax.block_until_ready(result)
        Numba: numba.cuda.synchronize
        CuPy : cp.cuda.runtime.deviceSynchronize
    """
    for _ in range(warmup):
        fn(*args)
        if sync_fn is not None: sync_fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        if sync_fn is not None: sync_fn()
        ts.append(time.perf_counter() - t0)
    return {
        "label":    label,
        "median_s": float(np.median(ts)),
        "min_s":    float(np.min(ts)),
        "max_s":    float(np.max(ts)),
        "repeats":  repeats,
    }


def save_timing(result, grid_str, tool, hardware, dtype,
                filepath=None, steps=None, **extra):
    """Append one timing record to filepath (a JSON list)."""
    path = Path(filepath if filepath is not None else TIMINGS_PATH)
    records = load_timings(str(path)) if path.exists() else []
    record = {
        "stage":    result.get("label", tool),
        "grid":     grid_str,
        "steps":    steps,
        "median_s": result["median_s"],
        "min_s":    result["min_s"],
        "max_s":    result["max_s"],
        "tool":     tool,
        "hardware": hardware,
        "dtype":    dtype,
    }
    record.update(extra)
    records.append(record)
    path.write_text(json.dumps(records, indent=2))


def load_timings(filepath=None):
    """Return all timing records as a list of dicts (empty if missing)."""
    path = Path(filepath if filepath is not None else TIMINGS_PATH)
    if not path.exists():
        return []
    return json.loads(path.read_text())


# --- Self-test --------------------------------------------------------------

def smoke(N=256, n_steps=100):
    """Run the bump-pulse IC for n_steps and assert the solution is sane."""
    L  = 10.0
    dx = L / N
    h, hu = bump_ic(N, L=L)                 # defaults h0=1.0, amplitude=0.1
    dt    = fixed_dt(1.1, dx)               # h_max = h0 + amplitude
    for _ in range(n_steps):
        apply_bc_reflective(h, hu)
        h, hu = step_numpy(h, hu, dx, dt)
    assert np.isfinite(h).all(),     "h is not finite"
    assert h.min() >= 0.0,            f"h.min() = {h.min()} < 0"
    centre = h[1 + N // 2]
    assert centre < 1.1 - 0.005,      "bump did not propagate out of the centre"
    print(f"swe_core OK  (N={N}, n_steps={n_steps}, dt={dt:.4e}, centre={centre:.4f})")


if __name__ == "__main__":
    smoke()
