# SPDX-License-Identifier: Apache-2.0
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

import json, os, time
from pathlib import Path
import numpy as np

g       = 9.81     # gravitational acceleration, m/s²
DRY_TOL = 1e-6     # depth below this is clamped (to avoid division by 0)

# As this is a canonical asset, allow arbitrary callers to resolve this file's location
HERE          = Path(__file__).resolve().parent
TIMINGS_PATH  = str(HERE / "timings.json")
SWE_STEP_CPP   = str(HERE / "swe_step.cpp")
SWE_CUB_CPP    = str(HERE / "swe_cub_solver.cpp")
SWE_RAW_CPP    = str(HERE / "swe_raw_cuda_solver.cpp")


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
    F_h  = 0.5 * (huL + huR) - 0.5 * a * (hR  - hL)
    F_hu = 0.5 * (huL*uL + 0.5*g*hL*hL + huR*uR + 0.5*g*hR*hR) - 0.5 * a * (huR - huL)

    # Divergence: update interior cells, ghost cells unchanged.
    h_new, hu_new = np.empty_like(h), np.empty_like(hu)
    h_new[0],  h_new[-1]  = h[0],  h[-1]
    hu_new[0], hu_new[-1] = hu[0], hu[-1]
    h_new[1:-1]   = h[1:-1]  - (dt / dx) * (F_h[1:]  - F_h[:-1])
    hu_new[1:-1]  = hu[1:-1] - (dt / dx) * (F_hu[1:] - F_hu[:-1])
    return h_new, hu_new


def solve_numpy(N, n_steps, L=10.0, h0=1.0, amplitude=0.1, sigma=0.5,
                cfl=0.4, g=g):
    """NumPy reference solver: bump IC, run n_steps of boundary conditions and update.
    Returns the final (h, hu)"""
    dx = L / N
    dt = fixed_dt(h0 + amplitude, dx, cfl=cfl, g=g)
    h, hu = bump_ic(N, L=L, h0=h0, amplitude=amplitude, sigma=sigma)
    for _ in range(n_steps):
        apply_bc_reflective(h, hu)
        h, hu = step_numpy(h, hu, dx, dt, g=g)
    return h, hu


# --- Validation -------------------------------------------------------------

def max_diff(a, b):
    """Max |a - b| as a float -  the cross-tool acceptance metric."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def report_and_verify(warm, diff, tol, cold_s=None,
                      n=None, steps=None, cold_note=""):
    """Verify results are within the defined error threshold, and emit timing and acceptance record.
    `warm`: dict returned by `timed_run`; `diff`: `max_diff(...)` against the fp64 NumPy reference.
    `cold_note`: append what the cold call includes. FAIL raises AssertionError
    """
    title = warm.get("label") or "run"
    ok   = diff < tol
    ctx  = f" N={n} steps={steps}" if n is not None and steps is not None else ""
    note = f" ({cold_note})" if cold_note else ""
    cold = f" | cold {cold_s * 1e3:7.1f} ms{note}" if cold_s is not None else ""
    record = (f"[{title}]{ctx}{cold} | warm {warm['median_s'] * 1e3:7.1f} ms"
              f" | max_diff {diff:.2e} {'<' if ok else '>='} tol {tol:.0e}"
              f" | {'PASS' if ok else 'FAIL'}")
    print(record)
    assert ok, record


# --- Timing harness ---------------------------------------------------------

def timed_run(fn, *args, warmup=2, repeats=5, label=""):
    """Time fn(*args) with warmups + repeats. Returns median / min / max [s].

    fn must not return until its work is done: every runner in this tutorial
    is responsible for handling synchronization and returns host arrays.
    """
    for _ in range(warmup):
        fn(*args)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return {
        "label":    label,
        "median_s": float(np.median(ts)),
        "min_s":    float(np.min(ts)),
        "max_s":    float(np.max(ts)),
        "repeats":  repeats,
        "samples_s": [float(t) for t in ts],
    }


def save_timing(result, grid_str, tool, hardware, dtype, steps=None, **extra):
    """Write a timing record to timings.json"""
    path = Path(TIMINGS_PATH)
    records = load_timings() if path.exists() else []
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
    # Keep one row per stage: a re-run replaces its previous entry.
    records = [r for r in records if r.get("stage") != record["stage"]]
    records.append(record)
    path.write_text(json.dumps(records, indent=2))


SWEEP_SIZES = ((16_384, 1000), (262_144, 500), (1_048_576, 200),
               (4_194_304, 100), (16_777_216, 50))


def timed_sweep(fn, warmup=1, repeats=3):
    """Rate fn(n_cells, n_steps) at every benchmark sweep point, for timings.json."""
    out = []
    for n, steps in SWEEP_SIZES:
        r = timed_run(fn, n, steps, warmup=warmup, repeats=repeats)
        out.append({"n": n, "steps": steps, "median_s": r["median_s"]})
    return out


def save_sweep(stage, fn, warmup=1, repeats=3):
    """Attach a benchmark sweep to an existing timings.json row.

    Run from the notebook that owns the stage, after its save_timing call:
    each tool is then measured in its own process, free of the others."""
    records = load_timings()
    row = next(r for r in records if r.get("stage") == stage)
    row["sweep"] = timed_sweep(fn, warmup=warmup, repeats=repeats)
    Path(TIMINGS_PATH).write_text(json.dumps(records, indent=2))


def breakdown_run(fn, n, steps, ic_s, transfer_s, warmup=1, repeats=3):
    """Split fn(n, steps)'s end-to-end median into components [s]: compute
    from the step-count slope, ic and transfers as measured by the caller,
    alloc_other as the remainder."""
    t1 = timed_run(fn, n, steps,     warmup=warmup, repeats=repeats)["median_s"]
    t2 = timed_run(fn, n, 2 * steps, warmup=warmup, repeats=repeats)["median_s"]
    compute = t2 - t1
    return {"total_s": t1, "compute_s": compute, "ic_s": ic_s,
            "transfer_s": transfer_s,
            "alloc_other_s": max(t1 - compute - ic_s - transfer_s, 0.0)}


def save_breakdown(stage, breakdown, n, steps):
    """Attach a `breakdown_run` result to an existing timings.json row.

    Like `save_sweep`, run from the notebook that owns the stage."""
    records = load_timings()
    row = next(r for r in records if r.get("stage") == stage)
    row["breakdown"] = {"n": n, "steps": steps, **breakdown}
    Path(TIMINGS_PATH).write_text(json.dumps(records, indent=2))


def load_timings():
    """Return all timing records as a list of dicts (empty if missing)."""
    path = Path(TIMINGS_PATH)
    return json.loads(path.read_text()) if path.exists() else []


# --- Experimental setup reports  ------------------------------------------

MACHINE_PATH = str(HERE / "machine.json")


def machine_report():
    """Document the measurement environment and experimental setup: software versions,
    CPU/GPU identity, clock/governor state."""
    import platform, subprocess
    info = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        info["cpu"] = [ln.split(":", 1)[1].strip() for ln in
                       open("/proc/cpuinfo") if ln.startswith("model name")][0]
        info["cores_physical_logical"] = "{}/{}".format(
            len({ln for ln in open("/proc/cpuinfo") if ln.startswith("core id")}) or "?",
            len([ln for ln in open("/proc/cpuinfo") if ln.startswith("processor")]))
    except OSError:
        pass
    try:
        info["cpu_governor"] = open(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except OSError:
        pass
    for mod in ("jax", "numba", "cupy"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
             "clocks.sm,clocks.max.sm,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if q.returncode == 0:
            (info["gpu"], info["driver"], info["gpu_mem"], info["gpu_sm_clock"],
             info["gpu_sm_clock_max"], info["gpu_temp_c"]) = \
                [s.strip() for s in q.stdout.strip().split(",")]
    except (OSError, subprocess.TimeoutExpired):
        pass
    w = max(len(k) for k in info)
    for k, v in info.items():
        print(f"  {k:<{w}}  {v}")
    return info


def measure_bandwidth_ceilings():
    """Run the standard bandwidth benchmarks: STREAM (CPU) and BabelStream (GPU)
    Saves best-rate roofs in GB/s to machine.json. Required binaries are installed in the tutorial image."""
    import shutil, subprocess

    def bench(cmd, **env):
        """Return {kernel: GB/s} from a STREAM-format results table."""
        out = subprocess.run([cmd], capture_output=True, text=True, timeout=300,
                             env={**os.environ, **env}).stdout
        return {p[0].rstrip(":"): round(float(p[1]) / 1e3, 1)
                for line in out.splitlines()
                if (p := line.split()) and p[0].rstrip(":") in ("Copy", "Triad")}

    ceilings = {}
    if shutil.which("stream_c"):
        cores = str(len({ln for ln in open("/proc/cpuinfo")
                         if ln.startswith("core id")}) or 1)
        ceilings["cpu_stream_triad_1t_gbs"] = bench("stream_c", OMP_NUM_THREADS="1")["Triad"]
        ceilings["cpu_stream_triad_gbs"] = bench(
            "stream_c", OMP_NUM_THREADS=cores,
            OMP_PROC_BIND="close", OMP_PLACES="cores")["Triad"]
    if shutil.which("cuda-stream"):
        gpu = bench("cuda-stream")
        ceilings["gpu_stream_triad_gbs"] = gpu["Triad"]
        ceilings["gpu_stream_copy_gbs"] = gpu["Copy"]
    for k, v in ceilings.items():
        print(f"  {k:<26} {v:8.1f} GB/s")
    if not ceilings:
        print("  stream_c / cuda-stream not found - run inside the tutorial image")
    else:
        Path(MACHINE_PATH).write_text(json.dumps(ceilings, indent=2))
    return ceilings


def to_mib(nbytes):
    """Bytes -> binary MiB (2**20 bytes)."""
    return nbytes / 2**20


def working_set_bytes(n_cells, n_arrays=6, itemsize=8):
    """Bytes one step of the canonical two-pass solver streams.

    h, hu double-buffered (4 state arrays) plus the two face-flux arrays.
    Fused single-pass kernels (e.g. PyOMP's) compute fluxes in registers
    and carry only the state: pass n_arrays=4."""
    return n_arrays * n_cells * itemsize


def llc_mib():
    """Largest data cache visible to core 0, in MiB (Linux sysfs)."""
    import glob
    sizes = []
    for p in glob.glob("/sys/devices/system/cpu/cpu0/cache/index*/size"):
        s = open(p).read().strip()
        sizes.append(float(s[:-1]) / 1024 if s.endswith("K") else float(s[:-1]) * 1024)
    return max(sizes) if sizes else None


def load_machine():
    """Return the ceilings measured by `measure_bandwidth_ceilings` ({} if absent)."""
    path = Path(MACHINE_PATH)
    return json.loads(path.read_text()) if path.exists() else {}


# --- Build helper -----------------------------------------------------------

def run_cmd(cmd, cwd=None):
    """Run a command, echo its last stdout line; raise with stderr on failure."""
    import subprocess
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}\n{r.stderr}")
    if r.stdout.strip():
        print(r.stdout.strip().splitlines()[-1])


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
