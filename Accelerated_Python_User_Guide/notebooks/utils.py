# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
"""Utility functions for validating FFT kernels in the 2D Navier-Stokes solver."""

import numpy as np
import warp as wp
import matplotlib.pyplot as plt


def validate_fft_roundtrip(
    fft_kernel,
    ifft_kernel,
    n_grid: int,
    tile_m: int,
    tile_n: int,
    block_dim: int,
    figsize=(14, 4),
):
    """Validate FFT and IFFT kernels using a sine wave test.

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
        inputs=[input_wp],
        outputs=[fft_result],
        block_dim=block_dim,
    )

    # Perform IFFT
    wp.launch_tiled(
        ifft_kernel,
        dim=[1, 1],
        inputs=[fft_result],
        outputs=[ifft_result],
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
    axes[0].plot(x, signal, "b-", linewidth=2, label="sin(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title("Original Signal: f(x) = sin(x)")
    axes[0].set_xlim([0, L])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel 2: FFT magnitude spectrum
    # Shift for better visualization (put k=0 at center)
    k_shifted = np.fft.fftshift(k)
    mag_shifted = np.fft.fftshift(fft_magnitude)

    axes[1].stem(k_shifted, mag_shifted, basefmt=" ")
    axes[1].set_xlabel("Wavenumber k")
    axes[1].set_ylabel(r"$|\hat{f}(k)|$")
    axes[1].set_title("FFT Amplitude Spectrum")
    axes[1].set_xlim([-10, 10])  # Focus on low wavenumbers
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=1, color="r", linestyle="--", alpha=0.5, label="k=1")
    axes[1].axvline(x=-1, color="r", linestyle="--", alpha=0.5, label="k=-1")
    axes[1].legend()

    # Panel 3: Comparison of original vs reconstructed
    axes[2].plot(x, signal, "b-", linewidth=2, label="Original")
    axes[2].plot(x, reconstructed, "r--", linewidth=2, label="FFT→IFFT")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("f(x)")
    axes[2].set_title(f"Comparison b/w original and reconstructed signal")
    axes[2].set_xlim([0, L])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    return fig, axes, max_error


def validate_transpose(
    transpose_kernel,
    tile_transpose_dim: int,
    n_test: int = 64,
    figsize=(10, 4),
):
    """Validate the tiled transpose kernel with a visual comparison.

    Creates an upper triangular matrix (1s above diagonal, 0s below),
    applies transpose, and shows before/after.

    Args:
        transpose_kernel: The tiled_transpose Warp kernel.
        tile_transpose_dim: TILE_TRANSPOSE_DIM parameter used in kernel.
        n_test: Size of the test matrix (default 64x64).
        figsize: Figure size tuple.

    Returns:
        fig, axes: Matplotlib figure and axes.
    """
    # Create upper triangular matrix: 1s above diagonal, 0s below
    pattern = np.triu(np.ones((n_test, n_test), dtype=np.float32))

    # Convert to complex format (wp.vec2f) - store pattern in real part
    input_complex = np.zeros((n_test, n_test, 2), dtype=np.float32)
    input_complex[:, :, 0] = pattern

    # Create Warp arrays
    input_wp = wp.array(input_complex, dtype=wp.vec2f, shape=(n_test, n_test))
    output_wp = wp.zeros((n_test, n_test), dtype=wp.vec2f)

    # Launch transpose kernel
    wp.launch_tiled(
        transpose_kernel,
        dim=(n_test // tile_transpose_dim, n_test // tile_transpose_dim),
        inputs=[input_wp],
        outputs=[output_wp],
        block_dim=tile_transpose_dim * tile_transpose_dim,
    )
    wp.synchronize()

    # Get result
    transposed_result = output_wp.numpy()[:, :, 0]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(pattern, cmap="gray_r")
    axes[0].set_title("Original")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")

    axes[1].imshow(transposed_result, cmap="gray_r")
    axes[1].set_title("After transpose")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")

    plt.tight_layout()
    return fig, axes


def validate_poisson_solver(
    solve_poisson_func,
    n_grid: int,
    kx: int = 1,
    ky: int = 1,
    figsize=(14, 4),
):
    """Validate the Poisson solver using an analytical test case.

    Tests with \omega = (kx^2 + ky^2)sin(kx·x)sin(ky·y), which has the analytical
    solution psi = sin(kx·x)sin(ky·y) for the Poisson equation \laplacian \psi = -\omega.

    Args:
        solve_poisson_func: The solve_poisson function to validate.
        n_grid: Grid resolution.
        kx: Wavenumber in x-direction (default 1).
        ky: Wavenumber in y-direction (default 1).
        figsize: Figure size tuple.

    Returns:
        fig, axes: Matplotlib figure and axes.
        max_error: Maximum absolute error between computed and analytical solution.
    """
    import warp as wp

    # Create 2D grid on [0, 2\pi) \times [0, 2\pi)
    L = 2.0 * np.pi
    x = np.linspace(0, L, n_grid, endpoint=False)
    y = np.linspace(0, L, n_grid, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Analytical solution: \psi = sin(kx·x)sin(ky·y)
    psi_analytical = np.sin(kx * X) * np.sin(ky * Y)

    # Corresponding vorticity: \omega = (kx^2 + ky^2)sin(kx·x)sin(ky·y)
    # From \laplacian\psi = -\omega, we have \omega = -\laplacian\psi = (kx^2 + ky^2)\psi
    k_squared = kx**2 + ky**2
    omega = k_squared * psi_analytical

    # Precompute 1/|k|^2 for spectral Poisson solver
    k_freq = np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    kx_grid, ky_grid = np.meshgrid(k_freq, k_freq)
    k2 = kx_grid**2 + ky_grid**2
    k2i_np = np.zeros_like(k2)
    nonzero = k2 != 0
    k2i_np[nonzero] = 1.0 / k2[nonzero]
    k2i = wp.array(k2i_np.astype(np.float32), dtype=wp.float32)

    # Create Warp arrays for omega and psi
    omega_wp = wp.array(omega.astype(np.float32), dtype=wp.float32)
    psi_wp = wp.zeros((n_grid, n_grid), dtype=wp.float32)

    # Solve Poisson equation
    solve_poisson_func(omega_wp, psi_wp, k2i)
    wp.synchronize()

    # Get computed solution
    psi_computed = psi_wp.numpy()

    # Compute error
    error = np.abs(psi_computed - psi_analytical)
    max_error = np.max(error)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Common colormap settings
    vmin = min(psi_analytical.min(), psi_computed.min())
    vmax = max(psi_analytical.max(), psi_computed.max())

    # Panel 1: Computed solution
    im0 = axes[0].imshow(
        psi_computed,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(r"Computed $\psi$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Panel 2: Analytical solution
    im1 = axes[1].imshow(
        psi_analytical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(r"Analytical $\psi = \sin(x)\sin(y)$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Panel 3: Error
    im2 = axes[2].imshow(error, extent=[0, L, 0, L], origin="lower", cmap="hot")
    axes[2].set_title(f"Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    return fig, axes, max_error


def validate_advection_diffusion(
    advection_diffusion_kernel,
    n_grid: int,
    kx: int = 2,
    ky: int = 3,
    figsize=(10, 4),
):
    """Validate finite-difference advection and diffusion using analytical test case.

    Uses test functions:
        omega = (kx^2 + ky^2) * sin(kx*x) * sin(ky*y)
        psi = sin(kx*x) * sin(ky*y)

    Analytical results:
        Diffusion: nabla^2 omega = -(kx^2 + ky^2)^2 * sin(kx*x) * sin(ky*y)
        Advection: J(psi, omega) = 0 (since omega = c * psi with constant c)

    Args:
        advection_diffusion_kernel: Warp kernel with signature:
            (omega, psi, advection_out, diffusion_out, h, n) where all arrays
            are wp.array2d(dtype=wp.float32), h is wp.float32, n is wp.int32.
        n_grid: Grid resolution.
        kx: Wavenumber in x-direction.
        ky: Wavenumber in y-direction.
        figsize: Figure size for each 1x2 panel.

    Returns:
        (fig_diff, axes_diff): Matplotlib figure/axes for diffusion comparison.
        (fig_adv, axes_adv): Matplotlib figure/axes for advection comparison.
        max_diff_error: Maximum absolute error in diffusion.
        max_adv_error: Maximum absolute error in advection.
    """
    # Create 2D grid on [0, 2*pi) x [0, 2*pi)
    L = 2.0 * np.pi
    h = L / n_grid
    x = np.linspace(0, L, n_grid, endpoint=False)
    y = np.linspace(0, L, n_grid, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Analytical fields
    k_squared = kx**2 + ky**2
    psi_analytical = np.sin(kx * X) * np.sin(ky * Y)
    omega_analytical = k_squared * psi_analytical

    # Analytical diffusion: nabla^2 omega = -(kx^2 + ky^2) * omega
    diffusion_analytical = -k_squared * omega_analytical

    # Analytical advection: J(psi, omega) = 0 (since omega = c * psi)
    advection_analytical = np.zeros_like(omega_analytical)

    # Create Warp arrays
    omega_wp = wp.array(omega_analytical.astype(np.float32), dtype=wp.float32)
    psi_wp = wp.array(psi_analytical.astype(np.float32), dtype=wp.float32)
    advection_wp = wp.zeros((n_grid, n_grid), dtype=wp.float32)
    diffusion_wp = wp.zeros((n_grid, n_grid), dtype=wp.float32)

    # Launch user-provided kernel
    wp.launch(
        advection_diffusion_kernel,
        dim=(n_grid, n_grid),
        inputs=[omega_wp, psi_wp, advection_wp, diffusion_wp, float(h), n_grid],
    )
    wp.synchronize()

    # Get numerical results
    diffusion_numerical = diffusion_wp.numpy()
    advection_numerical = advection_wp.numpy()

    # --- Diffusion comparison plot (1x2 panel) ---
    fig_diff, axes_diff = plt.subplots(1, 2, figsize=figsize)

    vmin_d = min(diffusion_analytical.min(), diffusion_numerical.min())
    vmax_d = max(diffusion_analytical.max(), diffusion_numerical.max())

    im0 = axes_diff[0].imshow(
        diffusion_analytical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_d,
        vmax=vmax_d,
    )
    axes_diff[0].set_title(r"Analytical $\nabla^2\omega$")
    axes_diff[0].set_xlabel("x")
    axes_diff[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes_diff[0], shrink=0.8)

    im1 = axes_diff[1].imshow(
        diffusion_numerical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_d,
        vmax=vmax_d,
    )
    axes_diff[1].set_title(rf"Numerical Diffusion")
    axes_diff[1].set_xlabel("x")
    axes_diff[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes_diff[1], shrink=0.8)

    fig_diff.suptitle(f"Diffusion Validation (kx={kx}, ky={ky}, N={n_grid})", y=1.02)
    fig_diff.tight_layout()

    # --- Advection comparison plot (1x2 panel) ---
    fig_adv, axes_adv = plt.subplots(1, 2, figsize=figsize)

    # For advection, analytical is zero, so set symmetric limits around numerical
    adv_max = max(np.abs(advection_numerical).max(), 1e-10)
    vmin_a, vmax_a = -adv_max, adv_max

    im2 = axes_adv[0].imshow(
        advection_analytical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_a,
        vmax=vmax_a,
    )
    axes_adv[0].set_title(r"Analytical Advection (= 0)")
    axes_adv[0].set_xlabel("x")
    axes_adv[0].set_ylabel("y")
    plt.colorbar(im2, ax=axes_adv[0], shrink=0.8)

    im3 = axes_adv[1].imshow(
        advection_numerical,
        extent=[0, L, 0, L],
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin_a,
        vmax=vmax_a,
    )
    axes_adv[1].set_title(rf"Numerical Advection")
    axes_adv[1].set_xlabel("x")
    axes_adv[1].set_ylabel("y")
    plt.colorbar(im3, ax=axes_adv[1], shrink=0.8)

    fig_adv.suptitle(f"Advection Validation (kx={kx}, ky={ky}, N={n_grid})", y=1.02)
    fig_adv.tight_layout()

    return (fig_diff, axes_diff), (fig_adv, axes_adv)
