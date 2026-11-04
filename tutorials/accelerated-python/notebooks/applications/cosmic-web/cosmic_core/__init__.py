# SPDX-License-Identifier: Apache-2.0
"""Shared code of the cosmic-web notebooks: the test field, the reference Euler characteristic, the benchmark harness
and the figures.

Every notebook imports the package as `cw`. The algorithm and its physics are explained in 90__cosmic_web__intro.ipynb.

config     constants and paths
device     device properties, free memory, the CuPy memory pool, the copy bandwidth
testfield  the Gaussian test field: spectrum, generation on the device, binning, cache
dataset    the density cube of notebooks 90 and 95: Quijote when its file is present, otherwise the Zel'dovich stand-in
ecc        the reference histogram, the Euler characteristic curve, the exact check, the Gaussian expectation
sizing     the memory budget of each stage, machine.json, the benchmark and sweep sizes, the copy bandwidth
bench      the first call and the median, sweeps, stage files, memory-traffic estimates
profile    profiler probes, Nsight Systems traces, Nsight Compute reports
kernels    the CUDA sources of notebook 94
plots      the figures and animations of notebooks 90 and 95
"""

import os

# CppJIT reads the interpreter arguments when `import cppjit` runs, so the optimisation level is set here, before it.
# The profiler probes rely on it: they `import cosmic_core` first.
os.environ.setdefault("CPPINTEROP_EXTRA_INTERPRETER_ARGS", "-O3")
del os

from .config import K_BINS
from .testfield import bin_edges, quantise, make_binned_field, field
from .dataset import load_dataset, zeldovich_particles
from .ecc import ecc_cells_cupy, ecc_hist_cupy, ecc_from_hist, chi_reference, chi_gaussian, zero_crossings, ANALYTIC_TOL
from .sizing import setup, save_sizing, benchmark_n, copy_gbs, machine_report
from .bench import (measure, save_sweep, by_size, load_stages, stage_table, wall_breakdown, sweep_series,
                    compile_report, traffic_report, call_table, compare)
from .profile import trace_call, ncu_report, JAX_HEADER
from .kernels import ecc_cuda_source, ecc_cuda_variants_source, cub_histogram_source
from .plots import (plot_slices, plot_hist_curve, plot_overlay, animate_particles, plot_points_to_field,
                    plot_birth_example, animate_flood, animate_euler_buildup, animate_birthdays, isosurfaces,
                    plot_first_call, plot_compile_share, plot_curves_with_delta, plot_timelines, plot_breakdown,
                    plot_traffic)
