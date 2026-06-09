# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

######################################################################################
# Example 2-D Navier-Stokes Optimization
#
# This example optimizes the initial vorticity field of a 2-D Navier-Stokes simulation
# so that after a specified number of timesteps, the vorticity field matches a target
# image (gtc_logo.png).
#
# The simulation uses the streamfunction-vorticity formulation with an FFT-based
# Poisson solver. Optimization is performed using a curriculum learning approach,
# starting with fewer timesteps and gradually increasing.
#
# Key differences from the pure simulation example:
# - All arrays are pre-allocated to avoid overwrites (required for autodiff)
# - Initial vorticity (omega_0) is the optimizable parameter
# - Loss is computed as normalized MSE between final vorticity and target image
#
######################################################################################

import os
import sys

import numpy as np

import warp as wp
import warp.optim

try:
    from PIL import Image
except ImportError as err:
    raise ImportError(
        "This example requires the Pillow package. Please install it with 'pip install Pillow'."
    ) from err

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# -----------------------------------------------------------------------------
# Simulation Constants
# -----------------------------------------------------------------------------

# Grid resolution (reduced for faster iteration during optimization)
N_GRID = 256

# Box size (2*pi for periodic domain)
LEN = 2 * np.pi

# Time step size
DT = 0.001

# Reynolds number
RE = 1000.0

# Vorticity range for target mapping
VORTICITY_MIN = -15.0
VORTICITY_MAX = 15.0

# Parameters for Warp's tiled-FFT functionality
TILE_M = 1
TILE_N = N_GRID
TILE_TRANSPOSE_DIM = 16
BLOCK_DIM = TILE_N // 2


# -----------------------------------------------------------------------------
# Warp helper functions
# -----------------------------------------------------------------------------

N_GRID_CONST = wp.constant(N_GRID)


@wp.func
def cyclic_index(idx: wp.int32, n: wp.int32) -> wp.int32:
    """Map any index to [0, n-1] for periodic boundary conditions."""
    ret_idx = idx % n
    if ret_idx < 0:
        ret_idx += n
    return ret_idx


@wp.func
def advection(
    omega_left: wp.float32,
    omega_right: wp.float32,
    omega_top: wp.float32,
    omega_down: wp.float32,
    psi_left: wp.float32,
    psi_right: wp.float32,
    psi_top: wp.float32,
    psi_down: wp.float32,
    h: wp.float32,
) -> wp.float32:
    """Calculate the advection term using central finite difference."""
    inv_2h = 1.0 / (2.0 * h)
    term_1 = ((omega_right - omega_left) * inv_2h) * ((psi_top - psi_down) * inv_2h)
    term_2 = ((omega_top - omega_down) * inv_2h) * ((psi_right - psi_left) * inv_2h)
    return term_2 - term_1


@wp.func
def diffusion(
    omega_left: wp.float32,
    omega_right: wp.float32,
    omega_center: wp.float32,
    omega_down: wp.float32,
    omega_top: wp.float32,
    h: wp.float32,
) -> wp.float32:
    """Calculate the Laplacian for viscous diffusion using central difference."""
    inv_h2 = 1.0 / (h * h)
    laplacian = (omega_right + omega_left + omega_top + omega_down - 4.0 * omega_center) * inv_h2
    return laplacian


# -----------------------------------------------------------------------------
# Autodiff-compatible Warp kernels
# -----------------------------------------------------------------------------


@wp.kernel
def compute_rhs_kernel(
    n: int,
    h: float,
    re: float,
    omega: wp.array2d(dtype=float),
    psi: wp.array2d(dtype=float),
    rhs: wp.array2d(dtype=float),
):
    """Compute the RHS of the vorticity transport equation (advection + diffusion).

    This kernel reads omega and psi, and writes to rhs.
    Separated from the RK update to avoid in-place modifications.
    """
    i, j = wp.tid()

    # Obtain neighboring indices with periodic BC
    left_idx = cyclic_index(i - 1, n)
    right_idx = cyclic_index(i + 1, n)
    top_idx = cyclic_index(j + 1, n)
    down_idx = cyclic_index(j - 1, n)

    # Compute viscous diffusion term
    diff_term = (1.0 / re) * diffusion(
        omega[left_idx, j],
        omega[right_idx, j],
        omega[i, j],
        omega[i, down_idx],
        omega[i, top_idx],
        h,
    )

    # Compute advection term
    adv_term = advection(
        omega[left_idx, j],
        omega[right_idx, j],
        omega[i, top_idx],
        omega[i, down_idx],
        psi[left_idx, j],
        psi[right_idx, j],
        psi[i, top_idx],
        psi[i, down_idx],
        h,
    )

    rhs[i, j] = diff_term + adv_term


@wp.kernel
def rk3_update_kernel(
    dt: float,
    coeff0: float,
    coeff1: float,
    coeff2: float,
    omega_0: wp.array2d(dtype=float),
    omega_in: wp.array2d(dtype=float),
    rhs: wp.array2d(dtype=float),
    omega_out: wp.array2d(dtype=float),
):
    """Perform RK3 update step: omega_out = coeff0*omega_0 + coeff1*omega_in + coeff2*dt*rhs.

    This kernel writes to omega_out, which must be a separate array from omega_in.
    """
    i, j = wp.tid()
    omega_out[i, j] = coeff0 * omega_0[i, j] + coeff1 * omega_in[i, j] + coeff2 * dt * rhs[i, j]


# -----------------------------------------------------------------------------
# FFT-related kernels
# -----------------------------------------------------------------------------


@wp.kernel
def copy_float_to_vec2(omega: wp.array2d(dtype=wp.float32), omega_complex: wp.array2d(dtype=wp.vec2f)):
    """Copy real vorticity to a complex array with zero imaginary part."""
    i, j = wp.tid()
    omega_complex[i, j] = wp.vec2f(omega[i, j], 0.0)


@wp.kernel(module="unique")
def fft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Perform 1-D FFT on each row using wp.tile_fft().

    Note: Uses module="unique" to isolate this kernel's compilation from other
    kernels. This prevents block_dim changes in other kernel launches from
    triggering recompilation with incompatible FFT configurations.
    """
    i, _, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_fft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel(module="unique")
def ifft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Perform 1-D inverse FFT on each row using wp.tile_ifft().

    Note: Uses module="unique" to isolate this kernel's compilation.
    """
    i, _, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_ifft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel(module="unique")
def tiled_transpose(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Transpose a 2-D array.

    Note: Uses module="unique" to isolate this kernel's compilation.
    """
    i, j = wp.tid()
    t = wp.tile_load(
        x,
        shape=(TILE_TRANSPOSE_DIM, TILE_TRANSPOSE_DIM),
        offset=(i * TILE_TRANSPOSE_DIM, j * TILE_TRANSPOSE_DIM),
        storage="shared",
    )
    t_transposed = wp.tile_transpose(t)
    wp.tile_store(y, t_transposed, offset=(j * TILE_TRANSPOSE_DIM, i * TILE_TRANSPOSE_DIM))


@wp.kernel
def multiply_k2_inverse(
    k2i: wp.array2d(dtype=wp.float32),
    omega_hat: wp.array2d(dtype=wp.vec2f),
    psi_hat: wp.array2d(dtype=wp.vec2f),
):
    """Solve Poisson equation in Fourier space: psi_hat = omega_hat / |k|^2."""
    i, j = wp.tid()
    psi_hat[i, j] = omega_hat[i, j] * k2i[i, j]


@wp.kernel
def extract_real_and_scale(
    scale: wp.float32,
    complex_array: wp.array2d(dtype=wp.vec2f),
    real_array: wp.array2d(dtype=wp.float32),
):
    """Extract real part from complex array and scale."""
    i, j = wp.tid()
    real_array[i, j] = complex_array[i, j].x / scale


# -----------------------------------------------------------------------------
# Loss computation kernel
# -----------------------------------------------------------------------------


@wp.kernel
def compute_loss_kernel(
    actual: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
):
    """Compute normalized MSE loss between actual vorticity and target."""
    i, j = wp.tid()

    diff = actual[i, j] - target[i, j]
    loss_value = diff * diff / wp.float32(N_GRID_CONST * N_GRID_CONST)

    wp.atomic_add(loss, 0, loss_value)


# -----------------------------------------------------------------------------
# Utility functions for target image loading
# -----------------------------------------------------------------------------


def load_target_image(image_path: str, grid_size: int, blur_sigma: float = 0.0) -> np.ndarray:
    """Load and preprocess target image for vorticity matching.

    Args:
        image_path: Path to the target image (should be black and white).
        grid_size: Target grid resolution to resize to.
        blur_sigma: Gaussian blur sigma for smooth gradients around edges.
            0 = no blur (sharp edges), higher values = smoother transitions.
            Recommended: 3-10 for smooth boundaries.

    Returns:
        NumPy array of shape (grid_size, grid_size) with vorticity values
        mapped from grayscale [0, 1] to [VORTICITY_MIN, VORTICITY_MAX].

    Note:
        The array is stored in simulation coordinates. When visualizing with
        `plt.imshow(arr.T, origin="lower")`, the image will appear correctly.
    """
    from scipy.ndimage import gaussian_filter

    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_resized = img.resize((grid_size, grid_size))

    # Convert to numpy and normalize to [0, 1] (like example_fluid_checkpoint)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # Apply Gaussian blur for smooth gradient transitions around letter edges
    if blur_sigma > 0:
        img_np = gaussian_filter(img_np, sigma=blur_sigma)
        # Re-normalize to ensure we use full [0, 1] range after blur
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)

    # Map to vorticity range:
    # black (0, background) -> VORTICITY_MIN
    # white (1, text interior) -> VORTICITY_MAX
    target_vorticity = VORTICITY_MIN + img_np * (VORTICITY_MAX - VORTICITY_MIN)

    # Transform to simulation coordinates:
    # - Image has y=0 at top, simulation visualization has y=0 at bottom
    # - Transpose so that image x becomes simulation's first index
    # Use .copy() to ensure we have an independent array (not a view)
    target_vorticity = np.flipud(target_vorticity).T.copy()

    return target_vorticity


def compute_target_average_vorticity(target: np.ndarray) -> float:
    """Compute the average vorticity of the target image."""
    return float(np.mean(target))


def create_initial_vorticity_with_bias(
    grid_size: int, target_avg: float, noise_scale: float = 0.1
) -> np.ndarray:
    """Create initial vorticity field biased to match target's average.

    For periodic boundary conditions, total vorticity is conserved.
    We initialize with small noise centered around the target average.

    Args:
        grid_size: Grid resolution.
        target_avg: Target average vorticity to match.
        noise_scale: Scale of random perturbations.

    Returns:
        Initial vorticity field as numpy array.
    """
    rng = np.random.default_rng(42)

    # Start with uniform field matching target average
    omega_init = np.full((grid_size, grid_size), target_avg, dtype=np.float32)

    # Add small random perturbations (zero-mean to preserve average)
    noise = rng.standard_normal((grid_size, grid_size)).astype(np.float32) * noise_scale
    noise -= noise.mean()  # Ensure zero-mean
    omega_init += noise

    return omega_init


# -----------------------------------------------------------------------------
# Main Example Class
# -----------------------------------------------------------------------------


class VorticityInverseSolver:
    """2-D Navier-Stokes optimization to match target vorticity distribution."""

    def __init__(
        self,
        target_image_path: str,
        lead_steps: int = 100,
        curriculum_increment: int = 10,
        n_grid: int = N_GRID,
        re: float = RE,
        dt: float = DT,
        length: float = LEN,
        blur_sigma: float = 3.0,
        use_cuda_graph: bool | None = False,
    ) -> None:
        """Initialize the optimization example.

        Args:
            target_image_path: Path to the target image.
            max_steps: Maximum number of simulation steps.
            curriculum_increment: Number of steps to add in each curriculum stage.
            n_grid: Grid resolution.
            re: Reynolds number.
            dt: Time step size.
            length: Physical domain size.
            blur_sigma: Gaussian blur for smooth edges (0=sharp, 3-10=smooth).
            use_cuda_graph: Enable CUDA graph capture for faster training.
                None = auto-detect (enable if on CUDA device).
        """
        self.n = n_grid
        self.length = length
        self.h = self.length / self.n
        self.re = re
        self.dt = dt
        self.max_steps = lead_steps
        self.curriculum_increment = curriculum_increment

        # SSP-RK3 coefficients: [coeff0, coeff1, coeff2]
        # omega_new = coeff0 * omega_0 + coeff1 * omega_current + coeff2 * dt * rhs
        self.rk3_coeffs = [
            [1.0, 0.0, 1.0],
            [3.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        ]

        # Load and process target image
        target_loaded = load_target_image(target_image_path, self.n, blur_sigma=blur_sigma)
        self.target_avg = compute_target_average_vorticity(target_loaded)

        # Create warp array for training
        self.target = wp.array2d(target_loaded, dtype=wp.float32)

        # Initialize optimizable initial condition
        omega_init_np = create_initial_vorticity_with_bias(self.n, self.target_avg, noise_scale=1.0)
        self.omega_init = wp.array2d(omega_init_np, dtype=wp.float32, requires_grad=True)

        # Pre-allocate arrays for forward pass
        self._allocate_arrays()

        # Loss array
        self.loss = wp.zeros((1,), dtype=float, requires_grad=True)

        # Optimizer (Adam)
        self.train_rate = 0.5
        self.optimizer = warp.optim.Adam([self.omega_init.flatten()], lr=self.train_rate)

        # Precompute 1/k^2 for spectral Poisson solver
        self._init_poisson_solver()

        # CUDA graph support (only works on CUDA devices)
        self.use_cuda_graph = use_cuda_graph and wp.get_device().is_cuda
        self.graph = None
        self.tape = None
        self.graph_num_steps = None  # Number of steps the current graph was captured for

        print(f"Initialized {self.n}x{self.n} solver (Re={self.re}, T={lead_steps} steps)")

    def _allocate_arrays(self) -> None:
        """Pre-allocate all arrays needed for the forward pass.

        For autodiff compatibility, we need separate arrays for each operation
        to avoid in-place modifications.
        """
        n = self.n
        num_steps = self.max_steps
        num_rk_stages = 3

        # SSP-RK3 time integration requires storing intermediate states.
        # We use separate arrays (no in-place ops) for autodiff compatibility.
        #
        # Time evolution: omega_timestep[0] -> ... -> omega_timestep[num_steps]
        #                 (initial condition)        (final state, compared to target)
        #
        # Within each timestep t, RK3 computes 3 stages:
        #   omega_stage[t][0], omega_stage[t][1], omega_stage[t][2]
        # The final stage result is copied to omega_timestep[t+1].

        self.omega_timestep = []
        self.omega_stage = []  # [timestep][stage] - intermediate RK3 vorticity
        self.psi_stage = []    # [timestep][stage] - streamfunction (from Poisson solve)
        self.rhs_stage = []    # [timestep][stage] - advection + diffusion terms

        for t in range(num_steps + 1):
            self.omega_timestep.append(wp.zeros((n, n), dtype=wp.float32, requires_grad=True))

        for t in range(num_steps):
            stage_omega = []
            stage_psi = []
            stage_rhs = []
            for s in range(num_rk_stages):
                stage_omega.append(wp.zeros((n, n), dtype=wp.float32, requires_grad=True))
                stage_psi.append(wp.zeros((n, n), dtype=wp.float32, requires_grad=True))
                stage_rhs.append(wp.zeros((n, n), dtype=wp.float32, requires_grad=True))
            self.omega_stage.append(stage_omega)
            self.psi_stage.append(stage_psi)
            self.rhs_stage.append(stage_rhs)

        # FFT temporary arrays for the spectral Poisson solver.
        # The solver computes psi from omega via: psi = FFT^{-1}(-omega_hat / k^2)
        #
        # Pipeline (each step needs a separate array for autodiff):
        #   omega_complex: real vorticity -> complex (re, 0)
        #   fft_temp_1:    after row-wise FFT
        #   fft_temp_2:    after transpose
        #   fft_temp_3:    after column-wise FFT (= omega_hat, full 2D transform)
        #   fft_temp_4:    after spectral division by -k^2
        #   (then IFFT back through fft_temp_1 -> fft_temp_2 -> fft_temp_3 -> psi)

        self.fft_arrays = []
        for t in range(num_steps):
            stage_fft = []
            for s in range(num_rk_stages):
                fft_temps = {
                    "omega_complex": wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                    "fft_temp_1": wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                    "fft_temp_2": wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                    "fft_temp_3": wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                    "fft_temp_4": wp.zeros((n, n), dtype=wp.vec2f, requires_grad=True),
                }
                stage_fft.append(fft_temps)
            self.fft_arrays.append(stage_fft)


    def _init_poisson_solver(self) -> None:
        """Initialize the Poisson solver (precompute 1/k^2)."""
        k = np.fft.fftfreq(self.n, d=1.0 / self.n)
        kx, ky = np.meshgrid(k, k)
        k2 = kx**2 + ky**2
        k2i = np.zeros_like(k2)
        nonzero = k2 != 0
        k2i[nonzero] = 1.0 / k2[nonzero]
        self.k2i = wp.array2d(k2i.astype(np.float32), dtype=wp.float32)

    def _solve_poisson(
        self, omega: wp.array2d, psi: wp.array2d, fft_temps: dict
    ) -> None:
        """Solve Poisson equation: nabla^2 psi = -omega using FFT.

        Uses pre-allocated FFT temporary arrays to avoid overwrites.
        """
        n = self.n

        # Convert omega to complex
        wp.launch(
            copy_float_to_vec2,
            dim=(n, n),
            inputs=[omega],
            outputs=[fft_temps["omega_complex"]],
        )

        # Forward 2D FFT (row FFT -> transpose -> row FFT)
        # Row FFT
        wp.launch_tiled(
            fft_tiled,
            dim=[n, 1],
            inputs=[fft_temps["omega_complex"]],
            outputs=[fft_temps["fft_temp_1"]],
            block_dim=BLOCK_DIM,
        )

        # Transpose
        wp.launch_tiled(
            tiled_transpose,
            dim=(n // TILE_TRANSPOSE_DIM, n // TILE_TRANSPOSE_DIM),
            inputs=[fft_temps["fft_temp_1"]],
            outputs=[fft_temps["fft_temp_2"]],
            block_dim=TILE_TRANSPOSE_DIM * TILE_TRANSPOSE_DIM,
        )

        # Column FFT (via row FFT on transposed)
        wp.launch_tiled(
            fft_tiled,
            dim=[n, 1],
            inputs=[fft_temps["fft_temp_2"]],
            outputs=[fft_temps["fft_temp_3"]],
            block_dim=BLOCK_DIM,
        )

        # Multiply by 1/k^2 in Fourier space
        wp.launch(
            multiply_k2_inverse,
            dim=(n, n),
            inputs=[self.k2i, fft_temps["fft_temp_3"]],
            outputs=[fft_temps["fft_temp_4"]],
        )

        # Inverse 2D FFT (row IFFT -> transpose -> row IFFT)
        # Row IFFT
        wp.launch_tiled(
            ifft_tiled,
            dim=[n, 1],
            inputs=[fft_temps["fft_temp_4"]],
            outputs=[fft_temps["fft_temp_1"]],
            block_dim=BLOCK_DIM,
        )

        # Transpose
        wp.launch_tiled(
            tiled_transpose,
            dim=(n // TILE_TRANSPOSE_DIM, n // TILE_TRANSPOSE_DIM),
            inputs=[fft_temps["fft_temp_1"]],
            outputs=[fft_temps["fft_temp_2"]],
            block_dim=TILE_TRANSPOSE_DIM * TILE_TRANSPOSE_DIM,
        )

        # Column IFFT
        wp.launch_tiled(
            ifft_tiled,
            dim=[n, 1],
            inputs=[fft_temps["fft_temp_2"]],
            outputs=[fft_temps["fft_temp_3"]],
            block_dim=BLOCK_DIM,
        )

        # Extract real part and normalize
        wp.launch(
            extract_real_and_scale,
            dim=(n, n),
            inputs=[float(n * n), fft_temps["fft_temp_3"]],
            outputs=[psi],
        )

    def forward(self, num_steps: int) -> None:
        """Run forward simulation for specified number of steps.

        Args:
            num_steps: Number of timesteps to simulate.
        """
        n = self.n

        # Copy initial condition to first timestep
        wp.copy(self.omega_timestep[0], self.omega_init)

        for t in range(num_steps):
            omega_t = self.omega_timestep[t]  # Vorticity at start of this timestep

            for s, (c0, c1, c2) in enumerate(self.rk3_coeffs):
                # Determine input omega for this stage
                if s == 0:
                    omega_in = omega_t
                else:
                    omega_in = self.omega_stage[t][s - 1]

                # Solve Poisson equation to get psi from omega
                self._solve_poisson(omega_in, self.psi_stage[t][s], self.fft_arrays[t][s])

                # Compute RHS (advection + diffusion)
                wp.launch(
                    compute_rhs_kernel,
                    dim=(n, n),
                    inputs=[n, self.h, self.re, omega_in, self.psi_stage[t][s]],
                    outputs=[self.rhs_stage[t][s]],
                )

                # RK3 update
                wp.launch(
                    rk3_update_kernel,
                    dim=(n, n),
                    inputs=[self.dt, c0, c1, c2, omega_t, omega_in, self.rhs_stage[t][s]],
                    outputs=[self.omega_stage[t][s]],
                )

            # Copy final stage result to next timestep's initial vorticity
            wp.copy(self.omega_timestep[t + 1], self.omega_stage[t][2])

        # Zero the loss before computing
        self.loss.zero_()

        # Compute loss: MSE between final vorticity and target
        wp.launch(
            compute_loss_kernel,
            dim=(n, n),
            inputs=[self.omega_timestep[num_steps], self.target],
            outputs=[self.loss],
        )

    def capture_graph(self, num_steps: int) -> None:
        """Capture a CUDA graph for the forward + backward pass.

        Args:
            num_steps: Number of simulation steps to capture.
        """
        if not self.use_cuda_graph:
            return

        with wp.ScopedCapture() as capture:
            self.tape = wp.Tape()
            with self.tape:
                self.forward(num_steps)
            self.tape.backward(self.loss)

        self.graph = capture.graph
        self.graph_num_steps = num_steps

    def train_step(self, num_steps: int) -> float:
        """Perform one training step (forward + backward + optimizer step).

        Args:
            num_steps: Number of simulation steps for this training iteration.

        Returns:
            Loss value.
        """
        if self.use_cuda_graph and self.graph is not None and self.graph_num_steps == num_steps:
            # Replay captured graph
            wp.capture_launch(self.graph)
        else:
            # Standard path (no graph or wrong number of steps)
            self.tape = wp.Tape()
            with self.tape:
                self.forward(num_steps)
            self.tape.backward(self.loss)

        # Optimizer step
        self.optimizer.step([self.omega_init.grad.flatten()])

        # Get loss value
        loss_val = self.loss.numpy()[0]

        # Zero gradients for next iteration
        self.tape.zero()

        return loss_val

    def run_curriculum_training(
        self, train_iters_per_stage: int = 50, verbose: bool = True
    ) -> None:
        """Run curriculum learning: gradually increase simulation steps.

        Args:
            train_iters_per_stage: Training iterations per curriculum stage.
            verbose: Print progress information.
        """
        num_stages = self.max_steps // self.curriculum_increment

        for stage in range(1, num_stages + 1):
            current_steps = stage * self.curriculum_increment

            if verbose:
                print(f"\n{'='*60}")
                print(f"Curriculum Stage {stage}/{num_stages}: {current_steps} simulation steps")
                print(f"{'='*60}")

            # Capture a new CUDA graph for this curriculum stage
            if self.use_cuda_graph:
                if verbose:
                    print("  Capturing CUDA graph...")
                self.capture_graph(current_steps)

            for iteration in range(train_iters_per_stage):
                loss = self.train_step(current_steps)

                if verbose and (iteration % 10 == 0 or iteration == train_iters_per_stage - 1):
                    print(f"  Iter {iteration:4d} | Loss: {loss:.6f}")

    def get_final_vorticity(self, num_steps: int) -> np.ndarray:
        """Run forward pass and return final vorticity as numpy array."""
        self.forward(num_steps)
        return self.omega_timestep[num_steps].numpy()

    def get_simulation_snapshots(self, num_steps: int, snapshot_interval: int = 1) -> list[np.ndarray]:
        """Run forward pass and return vorticity snapshots at regular intervals.

        Args:
            num_steps: Total number of simulation steps.
            snapshot_interval: Save a snapshot every N steps.

        Returns:
            List of numpy arrays containing vorticity at each snapshot time.
        """
        self.forward(num_steps)
        wp.synchronize()

        snapshots = []
        for t in range(0, num_steps + 1, snapshot_interval):
            snapshots.append(self.omega_timestep[t].numpy().copy())

        return snapshots


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="2-D Navier-Stokes Optimization to Target Image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the default Warp device.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="images/differentiable-navier-stokes/gtc_logo.png",
        help="Path to target image.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of simulation steps.",
    )
    parser.add_argument(
        "--curriculum_increment",
        type=int,
        default=100,
        help="Steps to add at each curriculum stage.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=100,
        help="Training iterations per curriculum stage.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing graphical windows.",
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=3.0,
        help="Gaussian blur sigma for smooth target edges (0=sharp, 5-10=smooth).",
    )
    parser.add_argument(
        "--use_cuda_graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA graph capture for faster training.",
    )

    args = parser.parse_known_args()[0]

    # Check visualization availability
    can_visualize = False
    if not args.headless:
        if not MATPLOTLIB_AVAILABLE:
            print(
                "Warning: matplotlib not found. Skipping visualization. "
                "Install matplotlib to enable: pip install matplotlib",
                file=sys.stderr,
            )
        elif matplotlib.get_backend().lower() == "agg":
            print(
                "Warning: No interactive matplotlib backend available. "
                "Skipping visualization.",
                file=sys.stderr,
            )
        else:
            can_visualize = True

    with wp.ScopedDevice(args.device):
        # Find target image path
        target_path = args.target
        if not os.path.exists(target_path):
            # Try looking in current directory or warp examples
            import warp.examples

            alt_path = os.path.join(os.path.dirname(__file__), args.target)
            if os.path.exists(alt_path):
                target_path = alt_path
            else:
                raise FileNotFoundError(f"Target image not found: {args.target}")

        example = VorticityInverseSolver(
            target_image_path=target_path,
            max_steps=args.max_steps,
            curriculum_increment=args.curriculum_increment,
            blur_sigma=args.blur_sigma,
            use_cuda_graph=args.use_cuda_graph,
        )

        # Report memory usage
        wp.synchronize_device()
        if (device := wp.get_device()).is_cuda:
            mem_gb = wp.get_mempool_used_mem_current(device) / (1024**3)
            print(f"GPU memory usage: {mem_gb:.3f} GiB")

        # Run curriculum training
        example.run_curriculum_training(train_iters_per_stage=args.train_iters)

        # Visualization - create animation of simulation evolution
        if can_visualize:
            print("\nGenerating simulation video...")

            # Get snapshots at regular intervals
            snapshot_interval = max(1, args.max_steps // 100)  # ~100 frames max
            snapshots = example.get_simulation_snapshots(args.max_steps, snapshot_interval)

            # Normalize vorticity to [0, 1] for visualization
            def normalize_vorticity(arr):
                return (arr - VORTICITY_MIN) / (VORTICITY_MAX - VORTICITY_MIN)

            # Create figure for video (simulation only, no borders)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_position([0, 0, 1, 1])  # Fill entire figure
            ax.axis("off")

            # Initialize image
            sim_img = ax.imshow(
                normalize_vorticity(snapshots[0].T),
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )

            def update(frame):
                """Update function for animation."""
                sim_img.set_data(normalize_vorticity(snapshots[frame].T))
                return [sim_img]

            # Create animation
            anim = FuncAnimation(
                fig,
                update,
                frames=len(snapshots),
                interval=50,  # 50ms between frames = 20 fps
                blit=True,
            )

            # Save as video
            video_path = "optimization_simulation.mp4"
            try:
                writer = FFMpegWriter(fps=20, metadata=dict(artist="Warp"), bitrate=1800)
                anim.save(video_path, writer=writer)
                print(f"Saved video to {video_path}")
            except Exception as e:
                print(f"Could not save MP4 (ffmpeg may not be installed): {e}")
                # Fallback to GIF
                gif_path = "optimization_simulation.gif"
                try:
                    anim.save(gif_path, writer="pillow", fps=20)
                    print(f"Saved GIF to {gif_path}")
                except Exception as e2:
                    print(f"Could not save GIF either: {e2}")

            if not args.headless:
                plt.show()
