"""
Test that CUDA tools (nvcc, nsys, ncu) work correctly.

Compiles a minimal CUDA C++ program using Thrust and verifies that both
Nsight Systems and Nsight Compute can successfully profile it.
"""

import pytest
import subprocess
from pathlib import Path

CUDA_PROGRAM = r"""
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <iostream>
#include <cstdlib>

int main() {
    constexpr int n = 256;

    thrust::device_vector<float> d(n);
    thrust::sequence(d.begin(), d.end());

    float sum = thrust::reduce(d.begin(), d.end());
    float expected = n * (n - 1) / 2.0f;

    if (sum != expected) {
        std::cerr << "Mismatch: got " << sum << ", expected " << expected << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "PASS" << std::endl;
}
"""


@pytest.fixture(scope="module")
def cuda_binary(tmp_path_factory):
    """Compile a minimal CUDA C++ program and return the path to the binary."""
    tmp_dir = tmp_path_factory.mktemp("nsight_test")
    src_path = tmp_dir / "test_program.cu"
    bin_path = tmp_dir / "test_program"

    src_path.write_text(CUDA_PROGRAM)

    result = subprocess.run(
        ["nvcc", "-o", str(bin_path), str(src_path)],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, \
        f"nvcc compilation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert bin_path.exists(), "Binary was not created"

    return bin_path


def test_cuda_binary_runs(cuda_binary):
    """Verify the compiled CUDA binary runs successfully."""
    result = subprocess.run(
        [str(cuda_binary)],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, \
        f"CUDA binary failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "PASS" in result.stdout


def test_nsys_profile(cuda_binary, tmp_path):
    """Test that nsys can profile the CUDA binary."""
    report_path = tmp_path / "test_report.nsys-rep"

    result = subprocess.run(
        ["nsys", "profile",
         "--force-overwrite=true",
         "--output", str(report_path),
         str(cuda_binary)],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, \
        f"nsys profile failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert report_path.exists(), "nsys report file was not created"


def test_ncu_profile(cuda_binary):
    """Test that ncu can profile the CUDA binary."""
    result = subprocess.run(
        ["ncu",
         "--target-processes=all",
         "--set=basic",
         str(cuda_binary)],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, \
        f"ncu profile failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
