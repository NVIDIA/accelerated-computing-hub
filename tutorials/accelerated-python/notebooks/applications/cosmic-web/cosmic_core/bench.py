# SPDX-License-Identifier: Apache-2.0
"""The benchmark harness: the first call and the median after the exact check, sweeps, stage files, memory-traffic
estimates."""

import inspect
import json
import re
import time

import numpy as np

from .config import (K_BINS, WARMUP, REPEATS, SWEEP_BUDGET_S, TIMINGS_DIR, STAGE_NOTEBOOK, RUN_INTRO_FIRST,
                     HELD_MEMORY_HINT)
from .device import free_bytes, release_pool
from .testfield import field, cache_path
from .ecc import gate
from .sizing import DEVICE_BYTES_PER_VOXEL, sweep_sizes, load_machine, copy_gbs


# --- Timing -----------------------------------------------------------------------

def timed_run(fn, *args):
    """{median_s, min_s, max_s, repeats, samples_s} of fn(*args) after WARMUP calls.

    `fn` must synchronise before it returns."""
    for _ in range(WARMUP):
        fn(*args)
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    release_pool()   # the pool holds the call's temporaries; every stage starts from the same free memory
    return {"median_s": float(np.median(ts)), "min_s": float(np.min(ts)), "max_s": float(np.max(ts)),
            "repeats": REPEATS, "samples_s": [float(t) for t in ts]}


def _gvoxel_s(n, seconds):
    """Voxels per second in units of 1e9."""
    return n ** 3 / seconds / 1e9


def measure(stage, run, n, *args, record=True, prepare=None, compile_s=0.0, note="", quiet=False):
    """Check the first call of run(n, *args) against the reference at size n, time it and the median of repeated calls,
    return the record.

    A failed check raises before any timing. `prepare(n)` runs before the timer, `compile_s` is added to the first call, `note` says what
    the first call includes, and with `record` the record is written to the stage file timings/<stage>.json."""
    field(n)
    if prepare is not None:
        prepare(n)
    t0 = time.perf_counter()
    hist = np.asarray(run(n, *args))
    cold_s = compile_s + time.perf_counter() - t0
    max_abs_diff, hist_sum = gate(n, hist)
    if max_abs_diff != 0 or hist_sum != 0:
        raise RuntimeError(f"[{stage}] N={n}^3 K={K_BINS}: max|Δχ| {max_abs_diff}, Σ {hist_sum}: the histogram does not "
                           "match the reference")
    timing = timed_run(run, n, *args)
    suffix = f" ({note})" if note else ""
    line = (f"[{stage}] N={n}^3 K={K_BINS} | first call {cold_s * 1e3:.1f} ms{suffix}"
            f" | median {timing['median_s'] * 1e3:.1f} ms = {_gvoxel_s(n, timing['median_s']):.3f} Gvoxel/s"
            f" | max|Δχ| {max_abs_diff}"
            f" | PASS")
    if not quiet:
        print(line)
    result = {"stage": stage, "n": n, "k": K_BINS, **timing, "cold_s": cold_s, "max_abs_diff": max_abs_diff,
              "hist_sum": hist_sum, "hist": hist.tolist()}
    if compile_s:
        result["compile_s"] = compile_s
    if record:
        TIMINGS_DIR.mkdir(exist_ok=True)
        (TIMINGS_DIR / f"{stage}.json").write_text(json.dumps(result, indent=2))
    return result


def timed_sweep(fn, sizes, bytes_per_voxel, first_call=False, prepare=None):
    """[{n, median_s, ...}] per sweep size, each checked against the reference before it is timed. Stops early and
    names the skipped sizes.

    The sweep stops at a size that would overrun SWEEP_BUDGET_S, or whose `bytes_per_voxel` (or the generator's, for a
    size not yet cached) the free device memory does not hold. With `first_call`, `prepare(n)` runs first and the first call is timed as `first_s`."""
    out, dropped, spent, reason = [], [], 0.0, ""
    sizes = list(sizes)
    for i, n in enumerate(sizes):
        if out:
            last = out[-1]
            per_run = last["median_s"] * (n / last["n"]) ** 3
            if spent + per_run * (WARMUP + REPEATS) > SWEEP_BUDGET_S:
                dropped, reason = sizes[i:], f"would overrun the {SWEEP_BUDGET_S:.0f} s budget"
                break
        # an uncached size is generated on the device first, at the generator's budget
        generate = 0 if cache_path("field", n, ".npz").exists() else DEVICE_BYTES_PER_VOXEL["generate"]
        need = max(bytes_per_voxel, generate) * n ** 3
        free = free_bytes()[1]
        if need > free:
            dropped = sizes[i:]
            reason = (f"needs {need / 2 ** 30:.1f} GiB of device memory and {free / 2 ** 30:.1f} GiB is free; "
                      f"{HELD_MEMORY_HINT}")
            break
        field(n)
        if prepare is not None:
            prepare(n)
        t0 = time.perf_counter()
        hist = fn(n)
        row = {"n": n}
        if first_call:
            row["first_s"] = time.perf_counter() - t0
        row["max_abs_diff"], row["hist_sum"] = gate(n, hist)
        if row["max_abs_diff"] != 0 or row["hist_sum"] != 0:
            raise RuntimeError(f"exact check failed at N={n}^3: {row}")
        r = timed_run(fn, n)
        spent += time.perf_counter() - t0
        row.update({"median_s": r["median_s"], "min_s": r["min_s"], "max_s": r["max_s"]})
        print(f"  N={n:>4}^3  " + (f"first call {row['first_s'] * 1e3:9.2f} ms  " if first_call else "")
              + f"median {r['median_s'] * 1e3:9.2f} ms  {_gvoxel_s(n, r['median_s']):7.3f} Gvoxel/s"
              f"  max|Δχ| {row['max_abs_diff']}")
        out.append(row)
        release_pool()
    if dropped:
        print(f"  sweep stopped before N = {dropped[0]}^3: it {reason}; skipped {', '.join(f'{m}^3' for m in dropped)}")
    return out


def by_size(build):
    """Memoise `build(n)`, so that what a runner needs at size n is built once.

    Pass the result as `prepare` to `measure` or `save_sweep`, or call it before timing, and no timed call pays for the
    setup. The getter keeps its defining line as `source` for the profiler probe, so write the `cw.by_size(...)` call on
    one line."""
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = build(n)
        return cache[n]

    context = inspect.getframeinfo(inspect.currentframe().f_back, context=1).code_context
    get.source = context[0].strip() if context else ""
    return get


# --- Stage files ------------------------------------------------------------------

def load_stage(stage):
    """The record in the stage file timings/<stage>.json; a missing file names the notebook to run first."""
    path = TIMINGS_DIR / f"{stage}.json"
    if not path.exists():
        notebook = STAGE_NOTEBOOK.get(stage, "the notebook that owns it")
        raise FileNotFoundError(f"no timings/{stage}.json yet: run {notebook} first")
    return json.loads(path.read_text())


def attach(stage, **fields):
    """Add `fields` to the stage file timings/<stage>.json."""
    record = load_stage(stage)
    record.update(fields)
    (TIMINGS_DIR / f"{stage}.json").write_text(json.dumps(record, indent=2))


def save_sweep(stage, run, first_call=False, prepare=None):
    """Measure the sweep over the sizes the stage affords, name the sizes it skips, and attach it to the stage file.

    With `first_call`, the first call at every size is timed too, after `prepare(n)`."""
    sizes = sweep_sizes(stage)
    rows = timed_sweep(run, sizes, DEVICE_BYTES_PER_VOXEL[stage], first_call, prepare)
    skipped = [int(m) for m in load_machine()["sweep"] if int(m) not in sizes]
    if skipped:
        per_voxel = DEVICE_BYTES_PER_VOXEL[stage]
        print(f"  skipped {', '.join(f'{m}^3' for m in skipped)}: {stage} needs {per_voxel} B/voxel, "
              f"{per_voxel * skipped[0] ** 3 / 2 ** 30:.0f} GiB at {skipped[0]}^3")
    attach(stage, sweep=rows)


def compile_report(stage, jaxpr, compiled):
    """Print and attach the jaxpr's primitives and the executable's fusions, scatters and temporaries."""
    lines = str(jaxpr).splitlines()
    rhs = [ln.split(" = ", 1)[1] for ln in lines if " = " in ln]
    # each right-hand side reads "name[params] args" or "name args"
    primitives = sorted({r.split("[")[0].split()[0] for r in rhs if not r.startswith("{")})
    print(f"jaxpr: {len(lines)} lines; primitives: {', '.join(primitives)}")
    text = compiled.as_text()
    n_fusion = len(re.findall(r", kind=k", text))   # every HLO fusion carries kind=kLoop, kInput or kOutput
    n_scatter = len(re.findall(r"\bscatter\(", text))
    temp_bytes = int(compiled.memory_analysis().temp_size_in_bytes)
    nbytes = load_stage(stage)["n"] ** 3 * 4
    print(f"compiled: {n_fusion} fusions, {n_scatter} scatters; temporaries {temp_bytes / nbytes:.1f} full-size arrays")
    attach(stage, compile={"fusions": n_fusion, "scatters": n_scatter, "temp_bytes": temp_bytes})


def load_stages(stages, extras=()):
    """The records of `stages` and `extras` for the synthesis, after its checks.

    Stops when machine.json or the `cupy` record is missing, or the records are at different sizes. A missing stage
    is named with its notebook and a missing extra is skipped; a record without its sweep or trace is named too."""
    machine = load_machine()
    if not machine.get("device_name"):
        raise RuntimeError(f"no machine.json: {RUN_INTRO_FIRST}")
    by_stage = ({p.stem: json.loads(p.read_text()) for p in sorted(TIMINGS_DIR.glob("*.json"))}
                if TIMINGS_DIR.exists() else {})
    if "cupy" not in by_stage:
        raise RuntimeError(f"no timing recorded for the reference: {RUN_INTRO_FIRST}, then this notebook")
    for s in stages:
        if s not in by_stage:
            print(f"stage {s!r} not recorded: run {STAGE_NOTEBOOK.get(s, s)} to add it")
    present = {s: by_stage[s] for s in (*stages, *extras) if s in by_stage}
    for s, r in present.items():
        missing = [k for k in ("sweep", "trace") if k not in r]
        if missing:
            print(f"stage {s!r} has no {' or '.join(missing)}: rerun {STAGE_NOTEBOOK.get(s, s)} top to bottom")
    sizes = {r["n"] for r in present.values()}
    if len(sizes) != 1:
        raise RuntimeError(f"stages were recorded at different sizes {sorted(sizes)}: rerun them on this machine")
    print(f"results recorded on {machine['device_name']}")
    return present


def stage_table(by_stage, labels):
    """Print one line per stage: size, bins, the median, the first call and the exact check."""
    print(f"\n{'tool':<15} {'N':>5} {'K':>4} {'median ms':>9} {'first call ms':>13} {'max|Δχ|':>8}")
    for s, r in by_stage.items():
        first = f"{r['cold_s'] * 1e3:13.1f}" if "cold_s" in r else f"{'-':>13}"
        print(f"{labels[s]:<15} {r['n']:>4}^3 {r['k']:>4} {r['median_s'] * 1e3:9.2f} {first} "
              f"{r.get('max_abs_diff', '?'):>8}")


def describe_call(stage, launches):
    """One line for a traced call: its stage, size, kernels, memcpys and memsets."""
    return (f"[{stage}] one call at N = {launches['n']}^3: kernels {launches['kernels']}, "
            f"memcpys {launches['memcpy']}, memsets {launches['memset']}")


def call_table(calls):
    """Print the kernels, memcpys and memsets of several traced calls side by side. `calls` maps a column label to a
    `trace_call` record."""
    labels = list(calls)
    sizes = {c["n"] for c in calls.values()}
    if len(sizes) != 1:
        raise RuntimeError(f"traces at different sizes: {sorted(sizes)}")
    width = max(12, *(len(label) + 2 for label in labels))
    print(f"{'one call at N = ' + str(sizes.pop()) + '^3':<36}" + "".join(f"{label:>{width}}" for label in labels))
    for name, key in (("kernels", "kernels"), ("memcpys", "memcpy"), ("memsets", "memset")):
        print(f"{name:<36}" + "".join(f"{calls[label][key]:>{width}}" for label in labels))


def compare(stage, baseline):
    """Print the stage's median against the baseline's, both read from their stage files."""
    a, b = load_stage(stage), load_stage(baseline)
    if a["n"] != b["n"]:
        raise RuntimeError(f"{stage} was recorded at {a['n']}^3 and {baseline} at {b['n']}^3: "
                           "rerun them on this machine")
    ratio = b["median_s"] / a["median_s"]
    verdict = f"{ratio:.1f} x faster" if ratio >= 1 else f"{1 / ratio:.1f} x slower"
    print(f"[{stage}] median {a['median_s'] * 1e3:.1f} ms against [{baseline}] {b['median_s'] * 1e3:.1f} ms: {verdict}")


def wall_breakdown(record):
    """{kernel_s, copies_s, other_s} of one call: kernels, copies and memsets, and the rest of the wall time."""
    rows, wall_s = record["trace"], record["launches"]["wall_s"]
    kernel = sum(d for _, d, _, kind in rows if kind == "kernel")
    copies = sum(d for _, d, _, kind in rows if kind != "kernel")
    return {"kernel_s": kernel, "copies_s": copies, "other_s": max(wall_s - kernel - copies, 0.0)}


def sweep_series(by_stage, labels):
    """Print the recorded sweeps as a table of Gvoxel/s per size and tool."""
    rate = {s: ([p["n"] for p in by_stage[s]["sweep"]],
                [_gvoxel_s(p["n"], p["median_s"]) for p in by_stage[s]["sweep"]])
            for s in labels}
    sizes = sorted({n for xs, _ in rate.values() for n in xs})
    print(f'{"N":>7}' + "".join(f"{labels[s]:>16}" for s in labels) + "   Gvoxel/s")
    for n in sizes:
        cells = [f"{ys[xs.index(n)]:>16.3f}" if n in xs else f'{"-":>16}' for xs, ys in rate.values()]
        print(f"{n:>4}^3 " + "".join(cells))


# --- Bytes models -----------------------------------------------------------------

def bytes_breakdown(stage, n):
    """[(operation, count, bytes)] of the DRAM traffic the stage's implementation issues at size n.

    The kernels' block flushes are reductions at L2, so their model is the one read of the field. A minimum over two
    views of one array reads it from DRAM once: the shifted view's lines are served by L1 and L2 (confirmed with
    Nsight Compute)."""
    v = n ** 3 * 4
    # bincount reads its input three times: the sign check, the maximum and the count. The sign check also writes and
    # reads a bool temporary: 0.5 of a float32 pass.
    if stage == "cupy":
        return [("roll (read + write)", 7, 2 * v),
                ("minimum (two reads + write)", 7, 3 * v),
                ("bincount (sign check, maximum, count)", 8, 3.5 * v)]
    if stage == "cupy_views":
        return [("minimum over views (read + write; the shifted view hits in cache)", 7, 2 * v),
                ("bincount (sign check, maximum, count)", 8, 3.5 * v)]
    if stage in ("jax", "jax_bincount"):
        return [("scatter operand (lower bound: one read of the field each)", 8, v)]
    if stage == "cccl":
        return [("histogram pass (one read of the field each)", 8, v)]
    if stage in ("numba", "cppjit_cuda"):
        return [("field (one read)", 1, v)]
    raise ValueError(f"no bytes model for {stage!r}")


def bytes_model(stage, n):
    """Total bytes of `bytes_breakdown` for the stage at size n."""
    return sum(count * nbytes for _, count, nbytes in bytes_breakdown(stage, n))


def traffic_report(stage, result):
    """Print the stage's memory-traffic estimate at the result's size: the table, then its time against the median."""
    n, median_s = result["n"], result["median_s"]
    rows = bytes_breakdown(stage, n)
    width = max(len(op) for op, _, _ in rows)
    print(f"{'operation':<{width}} {'count':>7} {'MiB each':>9} {'MiB total':>10}")
    for op, count, nbytes in rows:
        print(f"{op:<{width}} {count:>7} {nbytes / 2**20:>9.3f} {count * nbytes / 2**20:>10.1f}")
    gbs = copy_gbs()
    estimate = bytes_model(stage, n)
    estimate_s = estimate / (gbs * 1e9)
    print(f"estimate {estimate / 2**20:.0f} MiB -> {estimate_s * 1e3:.2f} ms at {gbs:.0f} GB/s; "
          f"measured {median_s * 1e3:.2f} ms = {median_s / estimate_s:.2f} x the estimate")
