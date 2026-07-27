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

os.environ.setdefault("CPPINTEROP_EXTRA_INTERPRETER_ARGS", "-O3")

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


# --- Problem sizing -----------------------------------------------------------
#
# Sizes are derived from the machine: core count, cache sizes, free memory.

CANONICAL_WORK = 200_000_000   # cells x steps in the headline run
SWEEP_WORK     = 200_000_000   # cells x steps at each sweep point
CELLS_PER_CORE = 15_000        # enough work per thread to amortise a fork/join
MIN_STEPS      = 20            # below this a step-rate measurement is noise
MAX_STEPS      = 1000


def _pow2_at_least(x):
    return max(1 << max(int(max(x, 1) - 1).bit_length(), 0), 1024)


def _pow2_nearest(x):
    up = _pow2_at_least(x)
    down = max(up // 2, 1024)
    return down if (x - down) < (up - x) else up


def canonical_size():
    """Headline (n_cells, n_steps): the value NB 08 recorded, else derived."""
    fixed = load_machine().get("canonical")
    return tuple(fixed) if fixed else _derive_canonical()


def _derive_canonical():
    """Headline (n_cells, n_steps): enough cells per core to amortise a
    parallel region, capped to keep the NumPy reference quick."""
    n = _pow2_nearest(physical_cores() * CELLS_PER_CORE)
    n = min(n, _pow2_at_least(_cell_budget()))
    steps = int(min(MAX_STEPS, max(MIN_STEPS, CANONICAL_WORK // n)))
    return n, steps


def _derive_sweep():
    """[(n_cells, n_steps)] spanning cache-resident to DRAM-resident.

    Runs from an eighth of the smaller cache boundary to eight times the
    larger one, with `n * steps` held near constant."""
    per_cell = working_set_bytes(1)
    boundaries = [m * 2**20 / per_cell
                  for m in (device_l2_mib(), llc_total_mib()) if m]
    lo = _pow2_at_least(min(boundaries) / 8) if boundaries else 16_384
    hi = _pow2_at_least(max(boundaries) * 8) if boundaries else 16_777_216
    hi = min(hi, _pow2_at_least(_cell_budget()))
    sizes, n = [], lo
    while n <= max(hi, lo):
        sizes.append((n, int(min(MAX_STEPS, max(MIN_STEPS, SWEEP_WORK // n)))))
        n *= 4
    return tuple(sizes)


def _cell_budget(fraction=0.4, bytes_per_cell=200):
    """Cells that fit in `fraction` of the tighter of host and device memory.

    The 200 B/cell allowance covers `step_numpy`'s temporaries."""
    host, dev = free_bytes()
    limits = [b for b in (host, dev) if b]
    return (min(limits) * fraction / bytes_per_cell) if limits else 16_777_216


def free_bytes():
    """(host, device) available memory in bytes; either is None if unknown."""
    host = dev = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    try:
        import cupy
        dev = int(cupy.cuda.runtime.memGetInfo()[0])
    except Exception:
        pass
    return host, dev


def device_l2_mib():
    """GPU L2 cache in MiB; None when no CUDA device is visible."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceProperties(0)["l2CacheSize"] / 2**20
    except Exception:
        return None


def llc_total_mib():
    """Last-level cache summed over every instance, in MiB; None if unknown."""
    import glob
    scale = {"K": 2**10, "M": 2**20, "G": 2**30}
    found = []
    for d in glob.glob("/sys/devices/system/cpu/cpu*/cache/index*/"):
        try:
            level = int(open(d + "level").read())
            shared = open(d + "shared_cpu_list").read().strip()
            size = open(d + "size").read().strip()
        except (OSError, ValueError):
            continue
        found.append((level, shared,
                      float(size.rstrip("KMG")) * scale.get(size[-1], 1) / 2**20))
    if not found:
        return None
    top = max(f[0] for f in found)
    return sum({shared: mib for level, shared, mib in found
                if level == top}.values())


_SWEEP_SIZES = None


def sweep_points():
    """Sweep points: the list NB 08 recorded, else derived once and cached."""
    fixed = load_machine().get("sweep")
    if fixed:
        return tuple(tuple(p) for p in fixed)
    global _SWEEP_SIZES
    if _SWEEP_SIZES is None:
        _SWEEP_SIZES = _derive_sweep()
    return _SWEEP_SIZES


def sweep_sizes():
    """Derive sweep points from the machine, ignoring any recorded list."""
    return _derive_sweep()


def __getattr__(name):
    # Deriving SWEEP_SIZES imports CuPy, so resolve it on first use.
    if name == "SWEEP_SIZES":
        return sweep_points()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def by_size(build):
    """Memoise `build(n_cells)` so a timed loop excludes the setup it returns.

    The warmup call pays for the setup; the timed calls reuse it."""
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = build(n)
        return cache[n]

    get.cache = cache
    return get


def timed_sweep(fn, warmup=1, repeats=3, budget_s=120.0):
    """Rate fn(n_cells, n_steps) at every sweep point, for timings.json.

    Every tool runs the identical (cells, steps) at each point. Do not vary the
    step count per tool: throughput is N / (fixed/steps + per_step), so fewer
    steps reads slower for a reason unrelated to the kernel.

    A tool that cannot reach the larger sizes inside `budget_s` stops early and
    the skipped sizes are named."""
    out, dropped, spent = [], [], 0.0
    points = sweep_points()
    for i, (n, steps) in enumerate(points):
        if out:
            last = out[-1]
            per_run = last["median_s"] * (n * steps) / (last["n"] * last["steps"])
            if spent + per_run * (warmup + repeats) > budget_s:
                dropped = [m for m, _ in points[i:]]
                break
        t0 = time.perf_counter()
        r = timed_run(fn, n, steps, warmup=warmup, repeats=repeats)
        spent += time.perf_counter() - t0
        out.append({"n": n, "steps": steps, "median_s": r["median_s"]})
    if dropped:
        print(f"  sweep stopped after N = {out[-1]['n']:,}; "
              f"{', '.join(f'{m:,}' for m in dropped)} would overrun the "
              f"{budget_s:.0f} s budget")
    return out


def sweep_table(runners, budget_s=60.0, warmup=1, repeats=3):
    """Rate every {label: fn(n_cells, n_steps)} at each sweep point, printing a
    row per size and returning {label: (n_cells, Mcells/s)}.

    A runner that has spent `budget_s` sits out the remaining sizes, so a slow
    tool cannot stall the sweep."""
    rates = {k: ([], []) for k in runners}
    spent = dict.fromkeys(runners, 0.0)
    for n, steps in sweep_points():
        row = []
        for name, fn in runners.items():
            if spent[name] > budget_s:
                continue
            t0 = time.perf_counter()
            r = timed_run(fn, n, steps, warmup=warmup, repeats=repeats)
            spent[name] += time.perf_counter() - t0
            rate = n * steps / r["median_s"] / 1e6
            rates[name][0].append(n)
            rates[name][1].append(rate)
            row.append(f"{name} {rate:7.0f}")
        print(f"N={n:>10,} steps={steps:>4}   " + "   ".join(row) + "   Mcells/s")
    return rates


def sweep_series(by_stage, names):
    """Read the recorded sweeps into {label: (n_cells, Mcells/s)}, print the
    table, and return (sorted sizes, series)."""
    rate = {names[s]: ([p["n"] for p in by_stage[s]["sweep"]],
                       [p["n"] * p["steps"] / p["median_s"] / 1e6
                        for p in by_stage[s]["sweep"]]) for s in names}
    sizes = sorted({n for xs, _ in rate.values() for n in xs})
    print(f'{"N":>11}' + "".join(f"{names[s]:>14}" for s in names) + "   Mcells/s")
    for n in sizes:
        cells = [f'{ys[xs.index(n)]:>14.0f}' if n in xs else f'{"-":>14}'
                 for xs, ys in (rate[names[s]] for s in names)]
        print(f"{n:>11,}" + "".join(cells))
    return sizes, rate


def save_sweep(stage, fn, warmup=1, repeats=3):
    """Attach a benchmark sweep to an existing timings.json row.

    Run from the notebook that owns the stage, after its save_timing call:
    each tool is then measured in its own process, free of the others."""
    records = load_timings()
    row = next(r for r in records if r.get("stage") == stage)
    row["sweep"] = timed_sweep(fn, warmup=warmup, repeats=repeats)
    Path(TIMINGS_PATH).write_text(json.dumps(records, indent=2))


def save_breakdown(stage, breakdown, n, steps):
    """Attach a `nsys_breakdown` result to an existing timings.json row.

    Like `save_sweep`, run from the notebook that owns the stage."""
    records = load_timings()
    row = next(r for r in records if r.get("stage") == stage)
    row["breakdown"] = {"n": n, "steps": steps, **breakdown}
    Path(TIMINGS_PATH).write_text(json.dumps(records, indent=2))


def nsys_breakdown(report, wall_s):
    """Split `wall_s` into GPU kernels, memory copies and host time [s].

    Kernel and copy times come from `nsys stats` on `report`, so the report must
    cover exactly the call that `wall_s` measured. Host time is the remainder:
    the interpreter, the launches and any wait the device did not fill."""
    import subprocess
    out = subprocess.run(
        ["nsys", "stats", "--force-export=true",
         "--report", "cuda_gpu_kern_sum",
         "--report", "cuda_gpu_mem_time_sum", str(report)],
        capture_output=True, text=True).stdout
    totals, section = {}, None
    for line in out.splitlines():
        if "(cuda_gpu_kern_sum)" in line:
            section = "kernel_s"
        elif "(cuda_gpu_mem_time_sum)" in line:
            section = "memcpy_s"
        elif section:
            f = line.split()
            if len(f) > 2 and f[0].replace(".", "", 1).isdigit():
                totals[section] = totals.get(section, 0) + int(f[1].replace(",", ""))
    kernel = totals.get("kernel_s", 0) / 1e9
    memcpy = totals.get("memcpy_s", 0) / 1e9
    return {"total_s": wall_s, "kernel_s": kernel, "memcpy_s": memcpy,
            "host_s": max(wall_s - kernel - memcpy, 0.0)}


def load_timings():
    """Return all timing records as a list of dicts (empty if missing)."""
    path = Path(TIMINGS_PATH)
    return json.loads(path.read_text()) if path.exists() else []


# --- Experimental setup reports  ------------------------------------------

MACHINE_PATH = str(HERE / "machine.json")


def physical_cores():
    """Physical core count, falling back to the logical count."""
    import psutil
    return psutil.cpu_count(logical=False) or psutil.cpu_count()


def machine_report():
    """Document the measurement environment and experimental setup: software versions,
    CPU/GPU identity, clock/governor state."""
    import platform, subprocess
    info = {"python": platform.python_version(), "numpy": np.__version__}
    # x86 kernels name the CPU in /proc/cpuinfo, aarch64 ones do not.
    try:
        info["cpu"] = next(
            (ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
             if ln.split(":", 1)[0].strip() in ("model name", "Model name", "Model")),
            platform.machine())
    except OSError:
        info["cpu"] = platform.machine()
    info["cores_physical_logical"] = f"{physical_cores()}/{os.cpu_count()}"
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
                [s.strip() for s in q.stdout.splitlines()[0].split(",")]
    except (OSError, subprocess.TimeoutExpired):
        pass
    w = max(len(k) for k in info)
    for k, v in info.items():
        print(f"  {k:<{w}}  {v}")
    return info


def save_sizing(extra=None):
    """Fix this machine's problem sizes and record them in machine.json.

    NB 08 settles them once, so every rung sweeps the same sizes."""
    record = dict(load_machine())
    record.update(extra or {})
    record["canonical"] = list(_derive_canonical())
    record["sweep"] = [list(p) for p in _derive_sweep()]
    Path(MACHINE_PATH).write_text(json.dumps(record, indent=2))
    n, steps = record["canonical"]
    print(f"  canonical                  N = {n:,} cells, {steps} steps")
    print("  sweep                      "
          + ", ".join(f"{m:,}" for m, _ in record["sweep"]))
    return record


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
    """Largest cache visible to core 0, in MiB; None if sysfs reports none."""
    import glob
    scale = {"K": 2**10, "M": 2**20, "G": 2**30}
    sizes = []
    for p in glob.glob("/sys/devices/system/cpu/cpu0/cache/index*/size"):
        s = open(p).read().strip()
        sizes.append(float(s.rstrip("KMG")) * scale.get(s[-1], 1) / 2**20)
    return max(sizes) if sizes else None


def load_machine():
    """Return machine.json's contents ({} if absent)."""
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


# --- Presentation helpers ---------------------------------------------------
#
# Cosmetic utility functions that receive measurements or derived quantities.

_MARKERS = ("o-", "s-", "d-", "v-", "^-", "D-", "p-", "x-", "+-")


def plot_rate_sweep(x, series, xlabel, title, ylabel="throughput [Mcells/s]",
                    boundaries=(), logy=True, from_zero=False):
    """Plot one line per named series against x on a log-x axis.

    boundaries is a sequence of (x, label) verticals, drawn where a caller-
    computed regime changes (e.g. cache to DRAM). A series may be a plain list
    of y values against the shared x, or its own (xs, ys) pair when a tool
    could not afford every size."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    series = {k: (v if isinstance(v, tuple) else (x[:len(v)], v))
              for k, v in series.items()}
    for (name, (xs, ys)), marker in zip(series.items(), _MARKERS):
        ax.plot(xs, ys, marker, label=name)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    elif from_zero:
        ax.set_ylim(0, max(max(ys) for _, ys in series.values()) * 1.15)
    for xb, label in boundaries:
        ax.axvline(xb, ls=":", color="#666", lw=1)
        ax.text(xb, ax.get_ylim()[0], f" {label}", rotation=90, va="bottom",
                ha="left", fontsize=8, color="#444")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def plot_cold_warm(labels, cold_s, warm_s, title):
    """Grouped cold-vs-warm bars on a log time axis."""
    import matplotlib.pyplot as plt
    x, w = np.arange(len(labels)), 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, [t * 1e3 for t in cold_s], w,
           label="cold (first call)", color="#c33")
    ax.bar(x + w / 2, [t * 1e3 for t in warm_s], w,
           label="warm (median)", color="#5a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set(ylabel="time [ms, log]", title=title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_compile_share(labels, cold_s, warm_s, title):
    """Stacked bars splitting each first call into execution and one-time compile."""
    import matplotlib.pyplot as plt
    warm_ms = [t * 1e3 for t in warm_s]
    comp_ms = [max(c - w, 0.0) * 1e3 for c, w in zip(cold_s, warm_s)]
    total = [w + c for w, c in zip(warm_ms, comp_ms)]
    ex_frac = [w / t for w, t in zip(warm_ms, total)]
    cm_frac = [c / t for c, t in zip(comp_ms, total)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, ex_frac, width=0.55, color="#27a", label="execution (time per run)")
    ax.bar(x, cm_frac, width=0.55, bottom=ex_frac, color="#c33",
           label="XLA compile (one-time)")
    for xi, (cf, warm) in enumerate(zip(cm_frac, warm_ms)):
        ax.text(xi, 1.02, f"{cf * 100:.0f}% compile\n{warm:,.0f} ms run",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, .25, .5, .75, 1.])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set(ylabel="share of the first-call time", title=title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def animate_pulse(n_cells=256, length=20.0, n_steps=2500, skip=40):
    """Animate the bump-pulse height over n_steps, as embedded HTML."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    dx = length / n_cells
    h, hu = bump_ic(n_cells, L=length)
    dt = fixed_dt(1.1, dx)
    frames = [(0, h[1:-1].copy())]
    for step in range(1, n_steps + 1):
        apply_bc_reflective(h, hu)
        h, hu = step_numpy(h, hu, dx, dt)
        if step % skip == 0:
            frames.append((step, h[1:-1].copy()))
    xs = (np.arange(n_cells) + 0.5) * dx
    fig, ax = plt.subplots(figsize=(8, 3.6))
    line, = ax.plot(xs, frames[0][1], color="#27a", linewidth=2)
    ax.set_xlim(0, length)
    ax.set_ylim(0.93, 1.13)
    ax.set(xlabel="x  [m]", ylabel="h  [m]",
           title="1D bump pulse - water height over time")
    ax.grid(alpha=0.3)
    stamp = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                    verticalalignment="top")

    def _update(idx):
        step, frame = frames[idx]
        line.set_ydata(frame)
        stamp.set_text(f"step {step:4d}   t = {step * dt:.3f} s")
        return line, stamp

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=80, blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def require_stages(by_stage, stages, key="sweep"):
    """Raise unless every stage carries `key`, naming the notebooks to re-run."""
    missing = [s for s in stages if key not in by_stage.get(s, {})]
    if missing:
        raise SystemExit(f"no {key} recorded for: {', '.join(missing)}."
                         " Re-run those notebooks, then this one.")


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
