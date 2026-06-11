# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
"""Warp kernels and utilities for the differentiable Navier-Stokes notebook.

Module-level constants are captured at compile time by tile-based kernels,
so changing N_GRID requires a kernel cache clear.
"""

import math

import numpy as np
import warp as wp

# ---------------------------------------------------------------------------
# Grid / simulation constants.
# ---------------------------------------------------------------------------
N_GRID = 256
LEN = 2.0 * math.pi
DT = 0.001
RE = 1000.0
VORTICITY_MIN = -15.0
VORTICITY_MAX = 15.0
H = LEN / N_GRID
INV_H2 = 1.0 / (H * H)
INV_2H = 1.0 / (2.0 * H)

# Tile parameters for FFT kernels
TILE_M = 1
TILE_N = N_GRID
BLOCK_DIM = TILE_N // 2


# ---------------------------------------------------------------------------
# Helper function
# ---------------------------------------------------------------------------

@wp.func
def cyclic_index(idx: wp.int32, n: wp.int32) -> wp.int32:
    """Map any index to [0, n-1] for periodic boundary conditions."""
    ret_idx = idx % n
    if ret_idx < 0:
        ret_idx += n
    return ret_idx


# ---------------------------------------------------------------------------
# Type-conversion kernels (SIMT)
# ---------------------------------------------------------------------------

@wp.kernel
def copy_float_to_vec2(
    omega: wp.array2d(dtype=wp.float32), omega_complex: wp.array2d(dtype=wp.vec2f)
):
    """Copy real field to complex array with zero imaginary part."""
    i, j = wp.tid()
    omega_complex[i, j] = wp.vec2f(omega[i, j], 0.0)


@wp.kernel
def extract_real_and_scale(
    scale: wp.float32,
    complex_array: wp.array2d(dtype=wp.vec2f),
    real_array: wp.array2d(dtype=wp.float32),
):
    """Extract real part from complex array and divide by scale factor."""
    i, j = wp.tid()
    real_array[i, j] = complex_array[i, j].x / scale


# ---------------------------------------------------------------------------
# Tile-based FFT kernels
# ---------------------------------------------------------------------------

@wp.kernel(module="unique")
def fft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Perform 1-D FFT on each row using wp.tile_fft()."""
    i, _, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_fft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel(module="unique")
def ifft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Perform 1-D inverse FFT on each row using wp.tile_ifft()."""
    i, _, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_ifft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel
def transpose(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Transpose a 2-D complex array: y[i, j] = x[j, i]."""
    i, j = wp.tid()
    y[i, j] = x[j, i]


# ---------------------------------------------------------------------------
# Fourier-space Poisson solve (SIMT)
# ---------------------------------------------------------------------------

@wp.kernel
def multiply_k2_inverse(
    k2i: wp.array2d(dtype=wp.float32),
    omega_hat: wp.array2d(dtype=wp.vec2f),
    psi_hat: wp.array2d(dtype=wp.vec2f),
):
    """Solve Poisson equation in Fourier space: psi_hat = omega_hat / |k|^2."""
    i, j = wp.tid()
    psi_hat[i, j] = omega_hat[i, j] * k2i[i, j]


# ---------------------------------------------------------------------------
# Poisson solve via FFT (host-side pipeline)
# ---------------------------------------------------------------------------

def solve_poisson(omega, psi, fft_temps, k2i):
    """Solve Poisson equation nabla^2 psi = -omega via FFT.

    Pipeline: real->complex, row FFT, transpose, row FFT, pointwise /|k|^2,
    row IFFT, transpose, row IFFT, extract real.

    Args:
        omega: Vorticity field (float32, N x N).
        psi: Stream function output (float32, N x N).
        fft_temps: Dict with keys 'omega_complex', 'fft_temp_1' .. 'fft_temp_4'
            (vec2f, N x N each).
        k2i: Precomputed 1/|k|^2 (float32, N x N).
    """
    n = N_GRID

    wp.launch(copy_float_to_vec2, dim=(n, n),
              inputs=[omega], outputs=[fft_temps["omega_complex"]])

    # Forward 2D FFT: row FFT -> transpose -> row FFT
    wp.launch_tiled(fft_tiled, dim=[n, 1],
                    inputs=[fft_temps["omega_complex"]],
                    outputs=[fft_temps["fft_temp_1"]],
                    block_dim=BLOCK_DIM)
    wp.launch(transpose, dim=(n, n),
              inputs=[fft_temps["fft_temp_1"]],
              outputs=[fft_temps["fft_temp_2"]])
    wp.launch_tiled(fft_tiled, dim=[n, 1],
                    inputs=[fft_temps["fft_temp_2"]],
                    outputs=[fft_temps["fft_temp_3"]],
                    block_dim=BLOCK_DIM)

    # Solve in Fourier space
    wp.launch(multiply_k2_inverse, dim=(n, n),
              inputs=[k2i, fft_temps["fft_temp_3"]],
              outputs=[fft_temps["fft_temp_4"]])

    # Inverse 2D FFT: row IFFT -> transpose -> row IFFT
    wp.launch_tiled(ifft_tiled, dim=[n, 1],
                    inputs=[fft_temps["fft_temp_4"]],
                    outputs=[fft_temps["fft_temp_1"]],
                    block_dim=BLOCK_DIM)
    wp.launch(transpose, dim=(n, n),
              inputs=[fft_temps["fft_temp_1"]],
              outputs=[fft_temps["fft_temp_2"]])
    wp.launch_tiled(ifft_tiled, dim=[n, 1],
                    inputs=[fft_temps["fft_temp_2"]],
                    outputs=[fft_temps["fft_temp_3"]],
                    block_dim=BLOCK_DIM)

    wp.launch(extract_real_and_scale, dim=(n, n),
              inputs=[float(n * n), fft_temps["fft_temp_3"]], outputs=[psi])


# ---------------------------------------------------------------------------
# Fused vorticity update (SIMT)
# ---------------------------------------------------------------------------

@wp.kernel
def viscous_advection_kernel(
    omega_in: wp.array2d(dtype=float),
    psi: wp.array2d(dtype=float),
    omega_out: wp.array2d(dtype=float),
):
    """Fused diffusion + advection + forward Euler step.

    Module-level constants N_GRID, RE, DT, INV_H2, and INV_2H are captured
    automatically by Warp.
    """
    i, j = wp.tid()

    left  = cyclic_index(i - 1, N_GRID)
    right = cyclic_index(i + 1, N_GRID)
    up    = cyclic_index(j + 1, N_GRID)
    down  = cyclic_index(j - 1, N_GRID)

    diff = (omega_in[left, j] + omega_in[right, j] + omega_in[i, up] + omega_in[i, down]
            - 4.0 * omega_in[i, j]) * INV_H2

    adv1 = ((omega_in[right, j] - omega_in[left, j]) * INV_2H
            * (psi[i, up] - psi[i, down]) * INV_2H)
    adv2 = ((omega_in[i, up] - omega_in[i, down]) * INV_2H
            * (psi[right, j] - psi[left, j]) * INV_2H)

    omega_out[i, j] = omega_in[i, j] + DT * (diff / RE - (adv1 - adv2))


# ---------------------------------------------------------------------------
# SSP-RK3 timestepping
# ---------------------------------------------------------------------------

RK3_COEFFS = [(1.0, 0.0, 1.0), (0.75, 0.25, 0.25), (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)]


@wp.kernel
def rk3_substep(
    c0: float,
    c1: float,
    c2: float,
    omega_0: wp.array2d(dtype=float),
    omega_in: wp.array2d(dtype=float),
    psi: wp.array2d(dtype=float),
    omega_out: wp.array2d(dtype=float),
):
    """One SSP-RK3 sub-stage: omega_out = c0*omega_0 + c1*omega_in + c2*dt*RHS.

    RHS = (1/Re)*laplacian(omega_in) - J(psi, omega_in).
    Module-level constants N_GRID, RE, DT, INV_H2, INV_2H are captured
    by Warp at compile time.
    """
    i, j = wp.tid()

    left = cyclic_index(i - 1, N_GRID)
    right = cyclic_index(i + 1, N_GRID)
    up = cyclic_index(j + 1, N_GRID)
    down = cyclic_index(j - 1, N_GRID)

    laplacian = (omega_in[right, j] + omega_in[left, j]
                 + omega_in[i, up] + omega_in[i, down]
                 - 4.0 * omega_in[i, j]) * INV_H2

    adv1 = ((omega_in[right, j] - omega_in[left, j]) * INV_2H
            * (psi[i, up] - psi[i, down]) * INV_2H)
    adv2 = ((omega_in[i, up] - omega_in[i, down]) * INV_2H
            * (psi[right, j] - psi[left, j]) * INV_2H)

    rhs = laplacian / RE - (adv1 - adv2)
    omega_out[i, j] = c0 * omega_0[i, j] + c1 * omega_in[i, j] + c2 * DT * rhs


@wp.kernel
def compute_loss_kernel(
    actual: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
):
    """Compute normalized MSE loss between actual and target vorticity."""
    i, j = wp.tid()
    diff = actual[i, j] - target[i, j]
    wp.atomic_add(loss, 0, diff * diff / wp.float32(N_GRID * N_GRID))


def rk3_step(omega_start, omega_curr, omega_next, psi, fft_temps, k2i):
    """One non-differentiable RK3 timestep with 3-buffer ping-pong.

    After return, omega_start holds the updated field.

    Args:
        omega_start: Vorticity at start of timestep (overwritten on return).
        omega_curr: Scratch buffer.
        omega_next: Scratch buffer.
        psi: Scratch buffer for stream function.
        fft_temps: Dict of FFT scratch arrays for solve_poisson.
        k2i: Precomputed 1/|k|^2.
    """
    n = N_GRID
    wp.copy(omega_curr, omega_start)
    for c0, c1, c2 in RK3_COEFFS:
        solve_poisson(omega_curr, psi, fft_temps, k2i)
        wp.launch(rk3_substep, dim=(n, n),
                  inputs=[c0, c1, c2, omega_start, omega_curr, psi],
                  outputs=[omega_next])
        omega_curr, omega_next = omega_next, omega_curr
    wp.copy(omega_start, omega_curr)


def compute_initial_state(n_spinup, t_lead, seed=42):
    """Spin-up to developed turbulence and produce unperturbed target.

    Runs n_spinup non-differentiable RK3 steps to reach a statistically
    stationary state (omega_0), then t_lead more steps to produce the
    unperturbed reference field (y_star).

    Args:
        n_spinup: Number of spin-up timesteps.
        t_lead: Number of unperturbed forward timesteps.
        seed: Random seed for IC generation.

    Returns:
        omega_0: Initial vorticity, shape (N_GRID, N_GRID), numpy float32.
        y_star: Unperturbed target at t_lead, shape (N_GRID, N_GRID), numpy float32.
        k2i: Precomputed 1/|k|^2, warp float32 array.
    """
    import numpy as np
    from helpers.navier_stokes.utils import initialize_decaying_turbulence

    n = N_GRID

    # Precompute 1/|k|^2
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k)
    k2 = kx**2 + ky**2
    k2i_np = np.zeros_like(k2)
    nonzero = k2 != 0
    k2i_np[nonzero] = 1.0 / k2[nonzero]
    k2i = wp.array(k2i_np.astype(np.float32), dtype=wp.float32, ndim=2)

    # Scratch buffers (freed when function returns)
    omega = wp.array(
        initialize_decaying_turbulence(n, seed=seed), dtype=wp.float32
    )
    omega_curr = wp.zeros((n, n), dtype=wp.float32)
    omega_next = wp.zeros((n, n), dtype=wp.float32)
    psi = wp.zeros((n, n), dtype=wp.float32)
    fft_temps = {
        "omega_complex": wp.zeros((n, n), dtype=wp.vec2f),
        "fft_temp_1":    wp.zeros((n, n), dtype=wp.vec2f),
        "fft_temp_2":    wp.zeros((n, n), dtype=wp.vec2f),
        "fft_temp_3":    wp.zeros((n, n), dtype=wp.vec2f),
        "fft_temp_4":    wp.zeros((n, n), dtype=wp.vec2f),
    }

    for _ in range(n_spinup):
        rk3_step(omega, omega_curr, omega_next, psi, fft_temps, k2i)
    wp.synchronize()
    omega_0 = omega.numpy().copy()

    for _ in range(t_lead):
        rk3_step(omega, omega_curr, omega_next, psi, fft_temps, k2i)
    wp.synchronize()
    y_star = omega.numpy().copy()

    return omega_0, y_star, k2i


# ---------------------------------------------------------------------------
# Target image loading and initial condition
# ---------------------------------------------------------------------------

def load_target_image(image_path, grid_size, blur_sigma=0.0):
    """Load target image and map grayscale to vorticity range.

    Args:
        image_path: Path to grayscale image.
        grid_size: Target grid resolution.
        blur_sigma: Gaussian blur sigma (0 = no blur).

    Returns:
        NumPy float32 array (grid_size, grid_size) in simulation coordinates.
        Visualize with ``plt.imshow(arr.T, origin="lower")``.
    """
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    img = Image.open(image_path).convert("L")
    img_resized = img.resize((grid_size, grid_size))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    if blur_sigma > 0:
        img_np = gaussian_filter(img_np, sigma=blur_sigma)
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)

    target = VORTICITY_MIN + img_np * (VORTICITY_MAX - VORTICITY_MIN)
    # Image y=0 at top, simulation y=0 at bottom; transpose so image-x -> sim index 0
    target = np.flipud(target).T.copy()
    return target


def create_initial_vorticity_with_bias(grid_size, target_avg, noise_scale=0.1):
    """Create initial vorticity biased to match target average.

    Periodic BCs conserve total vorticity, so we start near the target mean
    and add zero-mean noise for symmetry breaking.

    Args:
        grid_size: Grid resolution.
        target_avg: Mean vorticity of the target field.
        noise_scale: Amplitude of random perturbations.

    Returns:
        NumPy float32 array (grid_size, grid_size).
    """
    rng = np.random.default_rng(42)
    omega_init = np.full((grid_size, grid_size), target_avg, dtype=np.float32)
    noise = rng.standard_normal((grid_size, grid_size)).astype(np.float32) * noise_scale
    noise -= noise.mean()
    omega_init += noise
    return omega_init


# ---------------------------------------------------------------------------
# Differentiable solver class (for notebook Part 2)
# ---------------------------------------------------------------------------

class DifferentiableNSSolver:
    """Differentiable 2-D Navier-Stokes forward pass with SSA allocation.

    Encapsulates pre-allocated intermediate arrays and the forward simulation.
    Tape and optimizer are managed externally (shown inline in the notebook).
    """

    def __init__(self, target_np, omega_init_np, max_steps):
        """
        Args:
            target_np: Target vorticity, NumPy float32 (N_GRID, N_GRID).
            omega_init_np: Initial vorticity guess, NumPy float32 (N_GRID, N_GRID).
            max_steps: Maximum number of timesteps to allocate for.
        """
        n = N_GRID
        self.n = n
        self.max_steps = max_steps

        self.target = wp.array(target_np, dtype=wp.float32, ndim=2)
        self.omega_init = wp.array(omega_init_np, dtype=wp.float32, ndim=2,
                                   requires_grad=True)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        self._init_k2i()
        self._allocate_arrays()

    def _init_k2i(self):
        """Precompute 1/|k|^2 for the spectral Poisson solver."""
        k = np.fft.fftfreq(self.n, d=1.0 / self.n)
        kx, ky = np.meshgrid(k, k)
        k2 = kx**2 + ky**2
        k2i_np = np.zeros_like(k2)
        nonzero = k2 != 0
        k2i_np[nonzero] = 1.0 / k2[nonzero]
        self.k2i = wp.array(k2i_np.astype(np.float32), dtype=wp.float32, ndim=2)

    def _allocate_arrays(self):
        """Pre-allocate all SSA arrays for the differentiable forward pass.

        Layout (for T timesteps, 3 RK stages each):
          omega_timestep[T+1]      : vorticity at each timestep boundary
          omega_stage[T][3]        : intermediate RK3 vorticity
          psi_stage[T][3]          : stream function per stage
          fft_arrays[T][3][5 vec2f]: FFT scratch per Poisson solve

        No rhs_stage needed because rk3_substep is fused (RHS + update).
        """
        n = self.n
        T = self.max_steps

        self.omega_timestep = [
            wp.zeros((n, n), dtype=wp.float32, requires_grad=True) for _ in range(T + 1)
        ]

        self.omega_stage = []
        self.psi_stage = []
        self.fft_arrays = []
        for _ in range(T):
            stage_omega = [wp.zeros((n, n), dtype=wp.float32, requires_grad=True) for _ in range(3)]
            stage_psi = [wp.zeros((n, n), dtype=wp.float32, requires_grad=True) for _ in range(3)]
            stage_fft = []
            for _ in range(3):
                stage_fft.append({
                    "omega_complex": wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                    "fft_temp_1":    wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                    "fft_temp_2":    wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                    "fft_temp_3":    wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                    "fft_temp_4":    wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                })
            self.omega_stage.append(stage_omega)
            self.psi_stage.append(stage_psi)
            self.fft_arrays.append(stage_fft)

    def forward(self, num_steps):
        """Run differentiable forward simulation and compute MSE loss.

        Must be called inside a ``wp.Tape()`` context for gradient tracking.

        Args:
            num_steps: Number of timesteps to simulate (<= max_steps).
        """
        n = self.n

        wp.copy(self.omega_timestep[0], self.omega_init)

        for t in range(num_steps):
            omega_t = self.omega_timestep[t]

            for s, (c0, c1, c2) in enumerate(RK3_COEFFS):
                omega_in = omega_t if s == 0 else self.omega_stage[t][s - 1]

                solve_poisson(omega_in, self.psi_stage[t][s],
                              self.fft_arrays[t][s], self.k2i)
                wp.launch(rk3_substep, dim=(n, n),
                          inputs=[c0, c1, c2, omega_t, omega_in,
                                  self.psi_stage[t][s]],
                          outputs=[self.omega_stage[t][s]])

            wp.copy(self.omega_timestep[t + 1], self.omega_stage[t][2])

        self.loss.zero_()
        wp.launch(compute_loss_kernel, dim=(n, n),
                  inputs=[self.omega_timestep[num_steps], self.target],
                  outputs=[self.loss])


# ---------------------------------------------------------------------------
# GIF rendering
# ---------------------------------------------------------------------------

def save_optimization_gif(
    gif_snapshots,
    losses,
    omega_0,
    y_star,
    save_path="optimization.gif",
    dpi=120,
    duration_ms=200,
):
    """Render optimization progress as an animated GIF.

    Layout: loss curve on the left (spans 2 rows), 2x2 field panels on the
    right (omega_0, delta_omega, Y*, Y~). Color limits are fixed across all
    frames for visual stability.

    Args:
        gif_snapshots: List of (iter_idx, delta_omega_np, y_tilde_np) tuples.
        losses: Full loss history (list of floats, one per iteration).
        omega_0: Base vorticity field, shape (N, N), numpy float32.
        y_star: Unperturbed target field, shape (N, N), numpy float32.
        save_path: Output GIF path.
        dpi: Figure resolution.
        duration_ms: Frame duration in milliseconds.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from io import BytesIO
    from PIL import Image

    if not gif_snapshots:
        print("No GIF snapshots to render.")
        return

    # Fixed color limits across all frames
    omega_0_max = float(np.abs(omega_0).max())
    y_star_max = float(np.abs(y_star).max())
    dw_max = max(float(np.abs(snap[1]).max()) for snap in gif_snapshots)
    yt_max = max(float(np.abs(snap[2]).max()) for snap in gif_snapshots)

    # Loss axis limits with 5% padding
    loss_min, loss_max = float(np.min(losses)), float(np.max(losses))
    loss_pad = 0.05 * (loss_max - loss_min)
    loss_ylim = (loss_min - loss_pad, loss_max + loss_pad)

    pi_ticks = [0, np.pi, 2 * np.pi]
    pi_labels = [r"$0$", r"$\pi$", r"$2\pi$"]
    extent = [0, 2 * np.pi, 0, 2 * np.pi]

    fields_static = [omega_0, None, y_star, None]  # None = per-frame
    titles = [r"$\omega_0$", r"$\Delta\omega$", r"$Y^*$", r"$\tilde{Y}$"]
    vlims = [omega_0_max, dw_max, y_star_max, yt_max]
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]

    n_iters_total = len(losses)

    print(f"Rendering {len(gif_snapshots)} frames...")
    frames = []

    for iter_idx, dw_snap, yt_snap in gif_snapshots:
        fig = plt.figure(figsize=(10, 5), dpi=dpi)
        gs = GridSpec(2, 3, width_ratios=[1.2, 1, 1], figure=fig)

        # Loss curve (left, spans 2 rows)
        ax_loss = fig.add_subplot(gs[:, 0])
        ax_loss.plot(range(iter_idx + 1), losses[: iter_idx + 1], "k-", linewidth=1.0)
        ax_loss.plot(iter_idx, losses[iter_idx], "ro", markersize=4)
        ax_loss.set_xlim(0, n_iters_total - 1)
        ax_loss.set_ylim(*loss_ylim)
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel(r"$-\mathrm{MSE}(Y^*, \tilde{Y})$")
        ax_loss.set_title(f"Iteration {iter_idx}", fontsize=14, pad=8)
        ax_loss.grid(True, linewidth=0.3, alpha=0.5)

        # Field panels (2x2 on right)
        fields_frame = [omega_0, dw_snap, y_star, yt_snap]
        for (row, col), field, title, vlim in zip(positions, fields_frame, titles, vlims):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                field.T, origin="lower", cmap="RdBu_r",
                vmin=-vlim, vmax=vlim, extent=extent,
            )
            ax.set_title(title)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_xticks(pi_ticks)
            ax.set_xticklabels(pi_labels)
            ax.set_yticks(pi_ticks)
            ax.set_yticklabels(pi_labels)
            fig.colorbar(im, ax=ax, shrink=0.82)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF ({len(frames)} frames) to {save_path}")


def save_logo_optimization_gif(
    snapshots,
    losses,
    save_path="outputs/logo_optimization.gif",
    max_frames=30,
    dpi=100,
    duration_ms=200,
):
    """Render NVIDIA logo optimization progress as a GIF.

    Layout: [loss curve | evolving omega_0 | evolving omega_T]

    Subsamples snapshots to at most ``max_frames`` for a reasonable file size.
    Display inline with ``IPython.display.Image(filename=save_path)``.

    Args:
        snapshots: List of (iter_idx, omega_0_np, omega_T_np) tuples,
            one per optimization iteration. Arrays are (N, N) numpy float32
            in simulation coordinates.
        losses: Full loss history, one float per iteration.
        save_path: Output GIF path.
        max_frames: Target number of GIF frames (subsamples if needed).
        dpi: Figure resolution.
        duration_ms: Frame duration in milliseconds.
    """
    import os
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    if not snapshots:
        print("No snapshots to render.")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Subsample to ~max_frames, always including last
    n = len(snapshots)
    stride = max(1, n // max_frames)
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    selected = [snapshots[i] for i in indices]

    num_iters = len(losses)

    # Fixed symmetric color limits across all frames
    vmax_w0 = max(float(np.abs(s[1]).max()) for s in selected)
    vmax_wT = max(float(np.abs(s[2]).max()) for s in selected)
    extent = [0, 2.0 * np.pi, 0, 2.0 * np.pi]

    loss_min, loss_max = float(np.min(losses)), float(np.max(losses))
    loss_pad = 0.05 * (loss_max - loss_min)
    loss_ylim = (loss_min - loss_pad, loss_max + loss_pad)

    print(f"Rendering {len(selected)} frames (subsampled from {n})...")
    frames = []

    for iter_idx, w0, wT in selected:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi)

        # -- Left: loss curve --
        ax_loss = axes[0]
        ax_loss.plot(range(iter_idx + 1), losses[:iter_idx + 1], "k-", linewidth=1.0)
        ax_loss.plot(iter_idx, losses[iter_idx], "ro", markersize=5)
        ax_loss.set_xlim(0, num_iters - 1)
        ax_loss.set_ylim(*loss_ylim)
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("MSE Loss")
        ax_loss.set_title(f"Loss (iter {iter_idx})")
        ax_loss.grid(True, linewidth=0.3, alpha=0.5)

        # -- Middle: initial omega_0 (being optimized) --
        im_w0 = axes[1].imshow(
            w0.T, origin="lower", cmap="RdBu_r",
            vmin=-vmax_w0, vmax=vmax_w0, extent=extent,
        )
        axes[1].set_title(r"Initial $\omega_0$ (being optimized)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im_w0, ax=axes[1], shrink=0.82)

        # -- Right: final omega_T (target field from solver) --
        im_wT = axes[2].imshow(
            wT.T, origin="lower", cmap="RdBu_r",
            vmin=-vmax_wT, vmax=vmax_wT, extent=extent,
        )
        axes[2].set_title(r"Final $\omega_T$ (target field from solver)")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        fig.colorbar(im_wT, ax=axes[2], shrink=0.82)

        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF ({len(frames)} frames) to {save_path}")


def save_perturbation_optimization_gif(
    snapshots,
    losses,
    save_path="outputs/perturbation_optimization.gif",
    max_frames=30,
    dpi=100,
    duration_ms=200,
):
    """Render optimal-perturbation optimization progress as a GIF.

    Layout: [loss curve | evolving delta_omega | evolving omega_T]

    Subsamples snapshots to at most ``max_frames`` for a reasonable file size.
    Display inline with ``IPython.display.Image(filename=save_path)``.

    Args:
        snapshots: List of (iter_idx, delta_omega_np, omega_T_np) tuples,
            one per logged iteration. Arrays are (N, N) numpy float32
            in simulation coordinates.
        losses: Full loss history, one float per iteration.
        save_path: Output GIF path.
        max_frames: Target number of GIF frames (subsamples if needed).
        dpi: Figure resolution.
        duration_ms: Frame duration in milliseconds.
    """
    import os
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    if not snapshots:
        print("No snapshots to render.")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Subsample to ~max_frames, always including last
    n = len(snapshots)
    stride = max(1, n // max_frames)
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    selected = [snapshots[i] for i in indices]

    num_iters = len(losses)

    # Fixed symmetric color limits across all frames
    vmax_dw = max(float(np.abs(s[1]).max()) for s in selected)
    vmax_wT = max(float(np.abs(s[2]).max()) for s in selected)
    extent = [0, 2.0 * np.pi, 0, 2.0 * np.pi]

    loss_min, loss_max = float(np.min(losses)), float(np.max(losses))
    loss_pad = 0.05 * (loss_max - loss_min) if loss_max > loss_min else 0.1
    loss_ylim = (loss_min - loss_pad, loss_max + loss_pad)

    print(f"Rendering {len(selected)} frames (subsampled from {n})...")
    frames = []

    for iter_idx, dw, wT in selected:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi)

        # -- Left: loss curve --
        ax_loss = axes[0]
        ax_loss.plot(range(iter_idx + 1), losses[:iter_idx + 1], "k-", linewidth=1.0)
        ax_loss.plot(iter_idx, losses[iter_idx], "ro", markersize=5)
        ax_loss.set_xlim(0, num_iters - 1)
        ax_loss.set_ylim(*loss_ylim)
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Neg. MSE Loss")
        ax_loss.set_title(f"Loss (iter {iter_idx})")
        ax_loss.grid(True, linewidth=0.3, alpha=0.5)

        # -- Middle: perturbation delta_omega (being optimized) --
        im_dw = axes[1].imshow(
            dw.T, origin="lower", cmap="RdBu_r",
            vmin=-vmax_dw, vmax=vmax_dw, extent=extent,
        )
        axes[1].set_title(r"Perturbation $\Delta\omega$")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im_dw, ax=axes[1], shrink=0.82)

        # -- Right: perturbed field omega_T at lead time --
        im_wT = axes[2].imshow(
            wT.T, origin="lower", cmap="RdBu_r",
            vmin=-vmax_wT, vmax=vmax_wT, extent=extent,
        )
        axes[2].set_title(r"Perturbed $\tilde{\omega}_T$")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        fig.colorbar(im_wT, ax=axes[2], shrink=0.82)

        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved perturbation GIF ({len(frames)} frames) to {save_path}")
