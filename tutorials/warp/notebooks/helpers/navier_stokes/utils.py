# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
"""Utility functions for validating FFT kernels in the 2D Navier-Stokes solver."""

import numpy as np
import warp as wp
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})


def initialize_decaying_turbulence(n_grid: int, seed: int = 42) -> np.ndarray:
    """Initialize vorticity field for 2D decaying turbulence.

    Generates initial vorticity in Fourier space using the energy spectrum from
    San & Staples 2012, then transforms to physical space using IFFT.

    Args:
        n_grid: Grid resolution (N x N).
        seed: Random seed for reproducibility (default 42).

    Returns:
        omega: Initial vorticity field as numpy array of shape (n_grid, n_grid).
    """
    import math

    # Energy spectrum parameters (San & Staples 2012)
    k_p = 12.0  # Peak wavenumber
    s = 3  # Shape parameter

    # Compute wavenumber magnitudes
    k = np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    kx, ky = np.meshgrid(k, k)
    k_mag = np.sqrt(kx**2 + ky**2)

    # Compute energy spectrum E(k) following San & Staples 2012
    s_float = float(s)
    a_s = (2.0 * s_float + 1.0) ** (s_float + 1.0) / (2.0**s_float * math.factorial(s))

    # Avoid division by zero at k=0
    with np.errstate(divide="ignore", invalid="ignore"):
        k_ratio = k_mag / k_p
        energy_k = (
            (a_s / (2.0 * k_p))
            * k_ratio ** (2.0 * s_float + 1.0)
            * np.exp(-(s_float + 0.5) * k_ratio**2)
        )
        energy_k = np.nan_to_num(energy_k, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute amplitude from energy spectrum
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.sqrt(k_mag / np.pi * energy_k)
        amplitude = np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0)

    # Generate random phases with symmetry constraints for real-valued physical field
    rng = np.random.default_rng(seed)
    n_half = n_grid // 2 + 1
    zeta = 2.0 * np.pi * rng.random((n_half, n_half))
    eta = 2.0 * np.pi * rng.random((n_half, n_half))

    # Build full phase array with proper symmetry
    phase = np.zeros((n_grid, n_grid), dtype=np.float64)
    for i in range(n_grid):
        for j in range(n_grid):
            i_idx = i if i < n_half else n_grid - i
            j_idx = j if j < n_half else n_grid - j

            # Clamp indices to valid range
            i_idx = min(i_idx, n_half - 1)
            j_idx = min(j_idx, n_half - 1)

            if i < n_half and j < n_half:
                phase[i, j] = zeta[i_idx, j_idx] + eta[i_idx, j_idx]
            elif i >= n_half and j < n_half:
                phase[i, j] = -zeta[i_idx, j_idx] + eta[i_idx, j_idx]
            elif i >= n_half and j >= n_half:
                phase[i, j] = -zeta[i_idx, j_idx] - eta[i_idx, j_idx]
            else:
                phase[i, j] = zeta[i_idx, j_idx] - eta[i_idx, j_idx]

    # Construct omega_hat in Fourier space
    omega_hat = amplitude * (np.cos(phase) + 1j * np.sin(phase))

    # Transform to physical space (scale by N² since numpy's ifft2 normalizes by 1/N²)
    omega = (n_grid * n_grid * np.fft.ifft2(omega_hat)).real.astype(np.float32)

    return omega


def validate_fft_roundtrip(
    fft_kernel,
    ifft_kernel,
    n_grid: int,
    tile_m: int,
    tile_n: int,
    block_dim: int,
    figsize=(14, 4),
):
    r"""Validate FFT and IFFT kernels using a sine wave test.

    Creates a sin(x) function on [0, 2\pi], applies FFT to show the amplitude
    spectrum (should peak at k=1), then applies IFFT and compares with original.

    Args:
        fft_kernel: The fft_tiled Warp kernel.
        ifft_kernel: The ifft_tiled Warp kernel.
        n_grid: Grid resolution (must match TILE_N used in kernel compilation).
        tile_m: TILE_M parameter (typically 1 for row-wise FFT).
        tile_n: TILE_N parameter (typically N_GRID).
        block_dim: Block dimension for kernel launch.
        figsize: Figure size tuple.

    Returns:
        fig, axes: Matplotlib figure and axes.
        max_error: Maximum absolute error between original and reconstructed.
    """
    # Create physical grid on [0, 2\pi)
    L = 2.0 * np.pi
    x = np.linspace(0, L, n_grid, endpoint=False)

    # Create test signal: sin(x) which has a peak at k=1
    # Since f(x) = sin(x) = (e^{ix} - e^{-ix}) / 2i, FFT should show peaks at k=\pm 1
    signal = np.sin(x)

    # Prepare 2D array (fft_tiled operates on rows of 2D arrays)
    # We'll use a single row for this 1D test
    signal_2d = signal.reshape(1, n_grid).astype(np.float32)

    # Convert to complex (wp.vec2f format): [real, imag]
    # Shape must be (1, n_grid, 2) for Warp to interpret as (1, n_grid) of vec2f
    signal_complex = np.zeros((1, n_grid, 2), dtype=np.float32)
    signal_complex[:, :, 0] = signal_2d  # Real part
    signal_complex[:, :, 1] = 0.0  # Imaginary part = 0

    # Create Warp arrays - pass shape (1, n_grid) and let Warp handle vec2f
    input_wp = wp.array(signal_complex, dtype=wp.vec2f, shape=(1, n_grid))
    fft_result = wp.zeros((1, n_grid), dtype=wp.vec2f)
    ifft_result = wp.zeros((1, n_grid), dtype=wp.vec2f)

    # Perform FFT
    wp.launch_tiled(
        fft_kernel,
        dim=[1, 1],
        inputs=[input_wp, fft_result],
        block_dim=block_dim,
    )

    # Perform IFFT
    wp.launch_tiled(
        ifft_kernel,
        dim=[1, 1],
        inputs=[fft_result, ifft_result],
        block_dim=block_dim,
    )

    # Synchronize and get results
    wp.synchronize()

    # Convert back to numpy
    fft_np = fft_result.numpy()
    ifft_np = ifft_result.numpy()

    # Extract real and imaginary parts from FFT result
    fft_real = fft_np[0, :, 0]
    fft_imag = fft_np[0, :, 1]
    fft_magnitude = np.sqrt(fft_real**2 + fft_imag**2)

    # IFFT result (need to normalize by N)
    reconstructed = ifft_np[0, :, 0] / n_grid  # Real part, normalized

    # Wavenumbers for FFT (standard ordering: 0, 1, 2, ..., N/2-1, -N/2, ..., -1)
    k = np.fft.fftfreq(n_grid, d=L / (2 * np.pi * n_grid))

    # Compute error
    max_error = np.max(np.abs(signal - reconstructed))

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Original signal
    axes[0].plot(x, signal, "b-", linewidth=2, label=r"$\sin(x)$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$f(x)$")
    axes[0].set_title(r"Original Signal: $f(x) = \sin(x)$")
    axes[0].set_xlim([0, L])
    axes[0].set_xticks([0, np.pi, 2 * np.pi])
    axes[0].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel 2: FFT magnitude spectrum
    # Shift for better visualization (put k=0 at center)
    k_shifted = np.fft.fftshift(k)
    mag_shifted = np.fft.fftshift(fft_magnitude)

    axes[1].stem(k_shifted, mag_shifted, basefmt=" ")
    axes[1].set_xlabel(r"Wavenumber $k$")
    axes[1].set_ylabel(r"$|\hat{f}(k)|$")
    axes[1].set_title("FFT Amplitude Spectrum")
    axes[1].set_xlim([-10, 10])  # Focus on low wavenumbers
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=1, color="r", linestyle="--", alpha=0.5, label=r"$k=1$")
    axes[1].axvline(x=-1, color="r", linestyle="--", alpha=0.5, label=r"$k=-1$")
    axes[1].legend()

    # Panel 3: Comparison of original vs reconstructed
    axes[2].plot(x, signal, "b-", linewidth=2, label="Original")
    axes[2].plot(x, reconstructed, "r--", linewidth=2, label=r"FFT$\rightarrow$IFFT")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$f(x)$")
    axes[2].set_title("Comparison b/w original and reconstructed signal")
    axes[2].set_xlim([0, L])
    axes[2].set_xticks([0, np.pi, 2 * np.pi])
    axes[2].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    return fig, axes, max_error


def validate_transpose(
    transpose_kernel,
    n_test: int = 256,
    figsize=(10, 4),
):
    """Validate a SIMT transpose kernel with a visual comparison.

    Creates an upper triangular matrix (1s above diagonal, 0s below),
    applies transpose, and shows before/after. Upper triangular should
    become lower triangular after a correct transpose.

    Automatically detects in-place (single-array) vs out-of-place
    (two-array) kernels by inspecting the kernel's argument count.

    Args:
        transpose_kernel: A Warp kernel with signature (x,) for in-place
            or (x, y) for out-of-place transpose.
        n_test: Size of the test matrix (default 256x256).
        figsize: Figure size tuple.

    Returns:
        fig, axes: Matplotlib figure and axes.

    Raises:
        AssertionError: If an out-of-place kernel produces an incorrect transpose.
    """
    inplace = len(transpose_kernel.adj.args) == 1

    # Create upper triangular matrix: 1s above diagonal, 0s below
    pattern = np.triu(np.ones((n_test, n_test), dtype=np.float32))
    expected = pattern.T

    # Convert to complex format (wp.vec2f) - store pattern in real part
    input_complex = np.zeros((n_test, n_test, 2), dtype=np.float32)
    input_complex[:, :, 0] = pattern

    # Create Warp arrays
    input_wp = wp.array(input_complex, dtype=wp.vec2f, shape=(n_test, n_test))

    if inplace:
        wp.launch(transpose_kernel, dim=(n_test, n_test), inputs=[input_wp])
        wp.synchronize()
        transposed_result = input_wp.numpy()[:, :, 0]
    else:
        output_wp = wp.zeros((n_test, n_test), dtype=wp.vec2f)
        wp.launch(transpose_kernel, dim=(n_test, n_test), inputs=[input_wp, output_wp])
        wp.synchronize()
        transposed_result = output_wp.numpy()[:, :, 0]

    # Numerical check
    n_wrong = int(np.sum(transposed_result != expected))
    n_total = n_test * n_test
    passed = n_wrong == 0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(pattern, cmap="gray_r")
    axes[0].set_title("Original (upper triangular)")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")

    axes[1].imshow(transposed_result, cmap="gray_r")
    axes[1].set_title("After transpose (should be lower triangular)")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")

    plt.tight_layout()

    if passed:
        print(f"PASSED: all {n_total} elements match expected transpose.")
    else:
        print(f"FAILED: {n_wrong}/{n_total} elements differ from expected transpose.")

    if not inplace and not passed:
        raise AssertionError(
            f"Out-of-place transpose failed: {n_wrong}/{n_total} elements differ from the expected result."
        )

    return fig, axes


def validate_diffusion(
    diffusion_kernel,
    n_grid: int,
    kx: int = 2,
    ky: int = 3,
    figsize=(10, 4),
):
    """Validate finite-difference diffusion kernel using analytical test case.

    Uses test function:
        omega = sin(kx*x) * sin(ky*y)

    Analytical Laplacian:
        nabla^2 omega = -(kx^2 + ky^2) * sin(kx*x) * sin(ky*y)

    Args:
        diffusion_kernel: Warp kernel with signature:
            (omega, laplacian) where both are wp.array2d(dtype=wp.float32).
            Grid spacing and size are captured from module-level constants.
        n_grid: Grid resolution.
        kx: Wavenumber in x-direction.
        ky: Wavenumber in y-direction.
        figsize: Figure size for the 1x2 panel.

    Returns:
        (fig, axes): Matplotlib figure/axes for diffusion comparison.
    """
    # Create 2D grid on [0, 2*pi) x [0, 2*pi)
    L = 2.0 * np.pi
    h = L / n_grid
    x = np.linspace(0, L, n_grid, endpoint=False)
    y = np.linspace(0, L, n_grid, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Analytical fields
    k_squared = kx**2 + ky**2
    omega_analytical = np.sin(kx * X) * np.sin(ky * Y)
    laplacian_analytical = -k_squared * omega_analytical

    # Create Warp arrays
    omega_wp = wp.array(omega_analytical.astype(np.float32), dtype=wp.float32)
    laplacian_wp = wp.zeros((n_grid, n_grid), dtype=wp.float32)

    # Launch user-provided kernel
    wp.launch(
        diffusion_kernel,
        dim=(n_grid, n_grid),
        inputs=[omega_wp, laplacian_wp],
    )
    wp.synchronize()

    # Get numerical results
    laplacian_numerical = laplacian_wp.numpy()

    # Compute error
    max_error = np.max(np.abs(laplacian_analytical - laplacian_numerical))

    # --- Diffusion comparison plot (1x2 panel) ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    vmin_d = min(laplacian_analytical.min(), laplacian_numerical.min())
    vmax_d = max(laplacian_analytical.max(), laplacian_numerical.max())

    im0 = axes[0].imshow(
        laplacian_analytical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_d,
        vmax=vmax_d,
    )
    axes[0].set_title(r"Analytical $\nabla^2\omega$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_xticks([0, np.pi, 2 * np.pi])
    axes[0].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    axes[0].set_yticks([0, np.pi, 2 * np.pi])
    axes[0].set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(
        laplacian_numerical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_d,
        vmax=vmax_d,
    )
    axes[1].set_title(r"Numerical $\nabla^2\omega$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_xticks([0, np.pi, 2 * np.pi])
    axes[1].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    axes[1].set_yticks([0, np.pi, 2 * np.pi])
    axes[1].set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(
        f"Diffusion Validation ($k_x$={kx}, $k_y$={ky}, N={n_grid})", y=1.02
    )
    fig.tight_layout()

    return fig, axes


def validate_advection(
    advection_kernel,
    n_grid: int,
    kx: int = 2,
    ky: int = 3,
    figsize=(10, 4),
):
    """Validate finite-difference advection kernel using analytical test case.

    Uses test functions:
        psi = sin(kx*x) * sin(ky*y)
        omega = cos(kx*x) * cos(ky*y)

    Analytical advection:
        J(psi, omega) = (kx*ky/2) * (cos(2*kx*x) - cos(2*ky*y))

    Args:
        advection_kernel: Warp kernel with signature:
            (omega, psi, advection_out) where all arrays are
            wp.array2d(dtype=wp.float32). Grid spacing and size are captured
            from module-level constants.
        n_grid: Grid resolution.
        kx: Wavenumber in x-direction.
        ky: Wavenumber in y-direction.
        figsize: Figure size for the 1x2 panel.

    Returns:
        (fig, axes): Matplotlib figure/axes for advection comparison.
    """
    # Create 2D grid on [0, 2*pi) x [0, 2*pi)
    L = 2.0 * np.pi
    h = L / n_grid
    x = np.linspace(0, L, n_grid, endpoint=False)
    y = np.linspace(0, L, n_grid, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Analytical fields (chosen to give non-zero advection)
    psi_analytical = np.sin(kx * X) * np.sin(ky * Y)
    omega_analytical = np.cos(kx * X) * np.cos(ky * Y)

    # Analytical advection: J(psi, omega) = (kx*ky/2) * (cos(2*kx*x) - cos(2*ky*y))
    advection_analytical = (kx * ky / 2.0) * (
        np.cos(2 * kx * X) - np.cos(2 * ky * Y)
    )

    # Create Warp arrays
    omega_wp = wp.array(omega_analytical.astype(np.float32), dtype=wp.float32)
    psi_wp = wp.array(psi_analytical.astype(np.float32), dtype=wp.float32)
    advection_wp = wp.zeros((n_grid, n_grid), dtype=wp.float32)

    # Launch user-provided kernel
    wp.launch(
        advection_kernel,
        dim=(n_grid, n_grid),
        inputs=[omega_wp, psi_wp, advection_wp],
    )
    wp.synchronize()

    # Get numerical results
    advection_numerical = advection_wp.numpy()

    # Compute error
    max_error = np.max(np.abs(advection_analytical - advection_numerical))

    # --- Advection comparison plot (1x2 panel) ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    vmin_a = min(advection_analytical.min(), advection_numerical.min())
    vmax_a = max(advection_analytical.max(), advection_numerical.max())

    im0 = axes[0].imshow(
        advection_analytical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_a,
        vmax=vmax_a,
    )
    axes[0].set_title(r"Analytical $J(\psi, \omega)$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_xticks([0, np.pi, 2 * np.pi])
    axes[0].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    axes[0].set_yticks([0, np.pi, 2 * np.pi])
    axes[0].set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(
        advection_numerical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_a,
        vmax=vmax_a,
    )
    axes[1].set_title(r"Numerical $J(\psi, \omega)$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_xticks([0, np.pi, 2 * np.pi])
    axes[1].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    axes[1].set_yticks([0, np.pi, 2 * np.pi])
    axes[1].set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(
        f"Advection Validation ($k_x$={kx}, $k_y$={ky}, N={n_grid})", y=1.02
    )
    fig.tight_layout()

    return fig, axes
