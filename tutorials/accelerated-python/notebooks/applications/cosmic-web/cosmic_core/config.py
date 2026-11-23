# SPDX-License-Identifier: Apache-2.0
"""Constants of the problem and the paths of the generated files."""

from pathlib import Path

K_BINS          = 128
NU_LO, NU_HI    = -4.0, 4.0      # bin width 1/16, exact in binary
R_SMOOTH_VOXELS = 3.0
SEED            = 2026
SIGMA8          = 0.834
N_S, GAMMA      = 0.965, 0.21    # spectral index and shape parameter of the BBKS spectrum
CANONICAL_N     = 512            # the Quijote grid: the benchmark size of every tool
SWEEP_N         = (64, 128, 256, 512, 1024)
BOX_MPC         = 1000.0         # comoving box side in Mpc/h, the Quijote box
DEMO_BOX_MPC    = 250.0          # the intro's small Zel'dovich pictures
WARMUP, REPEATS = 2, 5
MAIN_STAGES     = ("cupy", "jax", "cccl", "numba", "cppjit_cuda")

# The notebook directory: the kernel sources live there, and data/, timings/ and machine.json are written there.
ROOT         = Path(__file__).resolve().parent.parent
TIMINGS_DIR  = ROOT / "timings"
MACHINE_PATH = ROOT / "machine.json"
DATA_DIR     = ROOT / "data"

STAGE_NOTEBOOK = {
    "cupy":         "90__cosmic_web__intro.ipynb",
    "cupy_views":   "91__cosmic_web__cupy__array_model.ipynb",
    "jax":          "92__cosmic_web__jax__fusion.ipynb",
    "jax_bincount": "92__cosmic_web__jax__fusion.ipynb",
    "cccl":         "93__cosmic_web__cccl__iterators.ipynb",
    "numba":        "93__cosmic_web__cccl__iterators.ipynb",
    "cppjit_cub":   "94__cosmic_web__cppjit__cuda_kernel.ipynb",
    "cppjit_cuda":  "94__cosmic_web__cppjit__cuda_kernel.ipynb",
}
RUN_INTRO_FIRST = f"run {STAGE_NOTEBOOK['cupy']} first"
HELD_MEMORY_HINT = ("device memory is held by another process, usually the Jupyter kernel of a notebook run before "
                    "this one: shut it down and run this cell again")
