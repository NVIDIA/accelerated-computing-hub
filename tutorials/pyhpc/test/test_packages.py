"""
Smoke tests for the pyhpc package stack.

Each test imports a library the tutorial relies on and exercises it just
enough to prove it is installed and functional on this machine (GPU, MPI,
and the C++/CUDA JIT toolchain included). These run fast and fail loudly,
so a broken image is caught before the much slower notebook suite.
"""

import subprocess
import sys
import warnings

import numpy as np


def test_cupy():
    """CuPy element-wise ops, reduction, and matmul on the GPU."""
    import cupy as cp

    a = cp.arange(10, dtype=cp.float64)
    assert float(cp.sum(a + a)) == 90.0
    m = cp.ones((4, 4))
    assert float(cp.matmul(m, m)[0, 0]) == 4.0


def test_numba_cuda():
    """numba.cuda JIT-compiles and runs an element-wise kernel on the GPU."""
    from numba import cuda

    @cuda.jit
    def add_one(x):
        i = cuda.grid(1)
        if i < x.size:
            x[i] += 1.0

    import cupy as cp

    x = cp.zeros(256, dtype=cp.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        add_one[1, 256](x)
        cuda.synchronize()
    assert float(cp.sum(x)) == 256.0


def test_cuda_cooperative():
    """cuda.cooperative builds a block-load algorithm (used by notebook 12)."""
    import cuda.coop as coop

    block_load = coop.block.load(np.uint8, 128, 4, "striped")
    assert block_load.files  # linkable device source was generated


def test_jax_gpu():
    """JAX runs a jit-compiled function on a CUDA device."""
    import jax
    import jax.numpy as jnp

    assert any(d.platform == "gpu" for d in jax.devices())

    @jax.jit
    def f(x):
        return jnp.sum(x * x)

    assert float(f(jnp.arange(5.0))) == 30.0


def test_pyomp():
    """PyOMP runs an OpenMP parallel-for region from an @njit function."""
    from numba.openmp import njit
    from numba.openmp import openmp_context as openmp

    @njit
    def parallel_sum(out, n):
        with openmp("parallel for"):
            for i in range(n):
                out[i] = i * 2

    out = np.zeros(1000, dtype=np.int64)
    parallel_sum(out, 1000)
    assert out[10] == 20
    assert int(out.sum()) == sum(i * 2 for i in range(1000))


def test_nanobind():
    """nanobind is importable and exposes its CMake support files."""
    import nanobind

    assert nanobind.cmake_dir()


def test_cppjit():
    """CppJIT compiles C++ in-process and calls it (CUDA enabled)."""
    import cppjit

    assert cppjit.CUDA_ENABLED, "CppJIT was built without CUDA support"
    cppjit.cppdef("int cppjit_smoke_add(int a, int b) { return a + b; }")
    assert cppjit.gbl.cppjit_smoke_add(2, 3) == 5


def test_cffi():
    """cffi declares and calls a C function from the system C library."""
    from cffi import FFI

    ffi = FFI()
    ffi.cdef("int abs(int);")
    libc = ffi.dlopen(None)
    assert libc.abs(-7) == 7


def test_memory_profiler():
    """memory_profiler samples the memory use of a callable."""
    from memory_profiler import memory_usage

    mem = memory_usage((sum, ([0] * 100_000,)))
    assert mem and max(mem) > 0


def test_nsightful():
    """nsightful imports and correctly reports a non-interactive context."""
    import nsightful

    assert nsightful.notebook.is_interactive_notebook() is False


def test_mpi4py():
    """mpi4py runs with multiple local ranks and reduces correctly."""
    program = (
        "from mpi4py import MPI\n"
        "comm = MPI.COMM_WORLD\n"
        "total = comm.allreduce(comm.Get_rank(), op=MPI.SUM)\n"
        "assert total == sum(range(comm.Get_size())), total\n"
        "if comm.Get_rank() == 0:\n"
        "    print('mpi4py ranks:', comm.Get_size())\n"
    )
    result = subprocess.run(
        ["mpirun.mpich", "-launcher", "fork", "-n", "4", sys.executable, "-c", program],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"mpirun failed:\n{result.stdout}\n{result.stderr}"
    assert "mpi4py ranks: 4" in result.stdout
