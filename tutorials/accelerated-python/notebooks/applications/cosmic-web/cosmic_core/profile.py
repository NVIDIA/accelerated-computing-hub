# SPDX-License-Identifier: Apache-2.0
"""Profiler probes: a stage's runner as a stand-alone script, traced by Nsight Systems or replayed by Nsight Compute."""

import csv
import inspect
import io
import subprocess
import sys
from pathlib import Path

from .config import HELD_MEMORY_HINT
from .bench import attach

PROBE_TEMPLATE = '''\
# Profiler probe for the stage {stage}: the notebook's functions, one warm call of the runner, then one call between
# cudaProfilerStart and cudaProfilerStop.
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the notebook directory
import numpy as np
import cupy as cp
import cosmic_core as cw
{header}
n = int(sys.argv[1])
K = cw.K_BINS

{impl}
{state}
call = lambda: {call}
call()
cp.cuda.get_current_stream().synchronize()
{prelude}cp.cuda.profiler.start()
t0 = time.perf_counter()
call()
cp.cuda.get_current_stream().synchronize()
wall = time.perf_counter() - t0
cp.cuda.profiler.stop()
print(f"WALL_S {{wall:.6f}}")
'''

# Before the profiled call, with `memory`: the pool is emptied, so the call allocates its temporaries afresh and the
# report records them.
PRELUDE_MEMORY = "cp.get_default_memory_pool().free_all_blocks()\n"

# The probe prelude of the JAX stages: the same environment and imports as the notebook's first cell.
JAX_HEADER = '''
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from functools import partial
import jax
import jax.numpy as jnp
'''

PROBE_DIR = Path("nsys-probes")      # the probe scripts
REPORT_DIR = Path("nsys-reports")    # the Nsight Systems reports and their exports

NSYS_PROFILE = ["nsys", "profile", "-c", "cudaProfilerApi", "--capture-range-end=stop", "--cuda-graph-trace=node",
                "--force-overwrite", "true"]


def write_probe(stage, run, functions=(), args=(), state=(), header="", memory=False):
    """Write nsys-probes/nsys_probe_<stage>.py, the stage's self-contained profiler script, and return its path.

    The script runs `header`, the source of `functions` and `run`, then the `state` lines (a line of code, or a
    `by_size` getter, whose defining line is copied), and calls `run(n, *args)` once warm and once profiled. With
    `memory`, the CuPy pool is emptied before the profiled call."""
    impl = "\n\n".join(inspect.getsource(inspect.unwrap(f)) for f in (*functions, run))
    lines = [x if isinstance(x, str) else x.source for x in state]
    call = f"{run.__name__}({', '.join(['n', *map(repr, args)])})"
    PROBE_DIR.mkdir(exist_ok=True)
    path = PROBE_DIR / f"nsys_probe_{stage}.py"
    script = PROBE_TEMPLATE.format(stage=stage, header=header, impl=impl, state="\n".join(lines), call=call,
                                   prelude=PRELUDE_MEMORY if memory else "")
    path.write_text(script)
    return str(path)


def run_probe(cmd):
    """The wall time in seconds of one run of a probe script.

    Raises with the output's tail when no WALL_S line appears."""
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    wall = next((float(ln.split()[1]) for ln in r.stdout.splitlines() if ln.startswith("WALL_S")), None)
    if r.returncode != 0 or wall is None:
        output = r.stdout + "\n" + r.stderr
        tail = "\n".join(output.splitlines()[-15:])
        hint = f"\nthe probe is a second process on the device: {HELD_MEMORY_HINT}" \
            if "out of memory" in output.lower() or "resource_exhausted" in output.lower() else ""
        command = " ".join(map(str, cmd))
        raise RuntimeError(f"probe produced no WALL_S line (exit {r.returncode}): {command}\n{tail}{hint}")
    return wall


# --- Nsight Systems ---------------------------------------------------------------

def nsys_trace_rows(report):
    """[(start_s, duration_s, name, kind)] per GPU operation of the report, with kind kernel, memcpy or memset."""
    out = subprocess.run(["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_trace",
                          "--format", "csv", str(report)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"nsys stats --report cuda_gpu_trace failed on {report}:\n{out.stderr[-2000:]}")
    lines = out.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Start (ns)" in ln), None)
    if start is None:
        raise RuntimeError(f"no 'Start (ns)' column in nsys stats --report cuda_gpu_trace for {report}")
    table = "\n".join(ln for ln in lines[start:] if ln.strip())
    rows = []
    for r in csv.DictReader(io.StringIO(table)):
        name = r.get("Name", "")
        low = name.lower()
        kind = "memcpy" if "memcpy" in low else "memset" if "memset" in low else "kernel"
        rows.append((int(r["Start (ns)"]) * 1e-9, int(r["Duration (ns)"]) * 1e-9, name, kind))
    if not rows:
        return rows
    t0 = min(s for s, *_ in rows)
    return [(s - t0, d, name, kind) for s, d, name, kind in rows]


def trace_call(stage, run, n, functions=(), args=(), state=(), header="", memory=False, quiet=False):
    """Write the stage's probe, trace one call with Nsight Systems, print and attach what the GPU did.

    Returns {"n", "kernels", "memcpy", "memset", "wall_s", "gpu_busy_s", "rows"}. With `memory`, the call allocates its
    temporaries afresh and the report records every allocation. With `quiet` nothing is printed.
    The report is nsys-reports/launch_<stage>.nsys-rep."""
    probe = write_probe(stage, run, functions, args, state, header, memory)
    options = NSYS_PROFILE + (["--cuda-memory-usage=true"] if memory else [])
    REPORT_DIR.mkdir(exist_ok=True)
    report = REPORT_DIR / f"launch_{stage}"
    wall_s = run_probe(options + ["-o", str(report), sys.executable, probe, str(n)])
    rows = nsys_trace_rows(report.with_suffix(".nsys-rep"))
    kinds = [kind for *_, kind in rows]
    launches = {"n": n, "kernels": kinds.count("kernel"), "memcpy": kinds.count("memcpy"),
                "memset": kinds.count("memset"), "wall_s": wall_s, "gpu_busy_s": sum(d for _, d, _, _ in rows)}
    if not quiet:
        print(f"[{stage}] one call at N = {n}^3: kernels {launches['kernels']}, memcpys {launches['memcpy']}, "
              f"memsets {launches['memset']}")
    attach(stage, launches=launches, trace=[list(r) for r in rows])
    return {**launches, "rows": rows}


# --- Nsight Compute ---------------------------------------------------------------

NCU_METRICS = {
    "gpu__time_duration.sum":                                           "kernel time",
    "sm__warps_active.avg.pct_of_peak_sustained_active":                "achieved occupancy (% of the SM's warp slots)",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":                 "SM throughput (% of peak)",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory throughput (% of peak)",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum":                   "global load sectors requested (32 B each)",
    "l1tex__t_sector_hit_rate.pct":                                     "L1 hit rate",
    "lts__t_sector_hit_rate.pct":                                       "L2 hit rate",
    "dram__bytes_read.sum":                                             "DRAM bytes read",
    "dram__bytes_write.sum":                                            "DRAM bytes written",
    "smsp__inst_executed_op_shared_atom.sum":                           "shared-memory atomics issued (warp level)",
    "lts__t_sectors_op_red.sum":                                        "L2 reduction sectors (32 B each)",
    "smsp__inst_executed.sum":                                          "instructions issued (warp level)",
    "smsp__issue_active.avg.pct_of_peak_sustained_active":              "issue slots used (%)",
}
NCU_EXTRA = ["lts__t_sectors_op_red.sum"]   # outside `--set full`, so requested by name


def ncu_profile(probe_cmd):
    """Raw metrics {name: (value, unit)} of the one launch `probe_cmd` brackets with cudaProfilerStart/Stop."""
    cmd = ["ncu", "--profile-from-start", "off", "--set", "full", "--metrics", ",".join(NCU_EXTRA),
           "--csv", "--page", "raw"]
    r = subprocess.run([str(c) for c in cmd + list(probe_cmd)], capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('"')]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"ncu produced no table (exit {r.returncode}):\n{(r.stdout + r.stderr)[-2000:]}")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    units = rows[0] if rows and rows[0].get("ID", "").strip() == "" else {}
    data = rows[1:] if units else rows
    if not data:
        raise RuntimeError("ncu reported no kernel")
    out = {}
    for key, value in data[-1].items():
        try:
            out[key] = (float(value.replace(",", "")), units.get(key, ""))
        except (ValueError, AttributeError):
            out[key] = (value, units.get(key, ""))
    return out


def ncu_report(stage, n, title, run=None, functions=(), args=(), state=(), header=""):
    """Print NCU_METRICS and the top stall reasons for one launch of the stage's probe, under `title`.

    With `run`, the probe is written first, as `trace_call` writes it."""
    if run is not None:
        write_probe(stage, run, functions, args, state, header)
    print(title)
    metrics = ncu_profile([sys.executable, str(PROBE_DIR / f"nsys_probe_{stage}.py"), str(n)])
    for key, label in NCU_METRICS.items():
        if key in metrics:
            value, unit = metrics[key]
            if isinstance(value, float):
                value = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.2f}"
            print(f"  {label:<52} {value:>14} {unit}")
    prefix, suffix = "smsp__average_warps_issue_stalled_", "_per_issue_active.ratio"
    stalls = {k[len(prefix):-len(suffix)]: v[0] for k, v in metrics.items()
              if k.startswith(prefix) and k.endswith(suffix) and isinstance(v[0], float)}
    # `selected` and `not_selected` are issue outcomes, not stall reasons
    stalls = {k: v for k, v in stalls.items() if not k.endswith("selected")}
    top = sorted(stalls.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print("  warps stalled per issue-active cycle, top reasons: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
