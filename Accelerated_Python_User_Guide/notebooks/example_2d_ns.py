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
# Example 2-D Navier-Stokes flow in a periodic box

# Implements a 2-D Navier Stokes solver in a periodic box using the
# streamfunction-vorticity formulation. The Poisson equation that relates streamfunction
# to vorticity is solved in Fourier space using tile-based FFT in Warp.
# Timestepping is performed using strong stability preserving RK3 scheme.

######################################################################################

from typing import Any, Optional

import numpy as np

import warp as wp

# grid resolution
N_GRID = 512
# box size
LEN = 2 * np.pi
# delta t for timestepping
DT = 0.001
# Reynolds number
RE = 1000.0

# parameters for Warp's tiled-FFT functionality
TILE_M = 1
TILE_N = N_GRID
TILE_TRANSPOSE_DIM = 16
BLOCK_DIM = TILE_N // 2


# -----------------------------------------------------------------------------
# Warp helper functions for flow initialization, periodicity imposition,
# calculating advection terms, and calculating diffusion components.
# -----------------------------------------------------------------------------


@wp.func
def factorial(n: wp.int32) -> wp.int32:
    """Compute factorial.

    Args:
        n: Input integer for which we want factorial.

    Returns:
        Factorial of input n.
    """
    result = wp.int32(1)
    for i in range(2, n + 1):
        result *= i
    return result


@wp.func
def energy_spectrum(k: wp.float32, s: wp.int32, kp: wp.float32) -> wp.float32:
    """Compute energy at wavenumber magnitude k.

    Follows San and Staples 2012 Computers and Fluids (page 49).
    https://www.sciencedirect.com/science/article/abs/pii/S0045793012001363.

    Args:
        k: Input wavenumber magnitude.
        s: Shape parameter of spectrum.
        kp: Wavenumber magnitude at which maximum of energy spectrum lies.

    Returns:
        Energy contained at wavenumber magnitude k.
    """
    s_factorial = wp.float32(factorial(s))
    s_float32 = wp.float32(s)
    a_s = (2.0 * s_float32 + 1.0) ** (s_float32 + 1.0) / (2.0**s_float32 * s_factorial)
    energy_k = (
        a_s
        / (2.0 * kp)
        * (k / kp) ** (2.0 * s_float32 + 1.0)
        * wp.exp(-(s_float32 + 0.5) * (k / kp) ** 2.0)
    )
    return energy_k


@wp.func
def phase_randomizer(
    n: int,
    zeta: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    i: int,
    j: int,
) -> wp.float32:
    """Calculate value of the random phase at index (i, j).

    Follows San and Staples 2012 to return phase value in any quadrant based on
    the values of eta and zeta in the first quadrant.

    Args:
        n: Size of the simulation domain.
        zeta: First phase function.
        eta: Second phase function
        i: rowwise index on the 2-D simulation domain.
        j: columnwise index on the 2-D simulation domain

    Returns:
        Value of the random phase in any quadrant.
    """
    n_half = n // 2

    # first quadrant
    if i < n_half and j < n_half:
        return zeta[i, j] + eta[i, j]
    # second quadrant
    if i >= n_half and j < n_half:
        return -zeta[n - i, j] + eta[n - i, j]
    # third quadrant
    if i >= n_half and j >= n_half:
        return -zeta[n - i, n - j] - eta[n - i, n - j]
    # fourth quadrant
    return zeta[i, n - j] - eta[i, n - j]


@wp.func
def cyclic_index(idx: wp.int32, n: wp.int32) -> wp.int32:
    """Map any index to [0, n-1] for periodic boundary conditions.

    Args:
        idx: Input index that may be outside the valid range.
        n: Grid size defining the periodic domain.

    Returns:
        Index wrapped to the range [0, n-1].
    """
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
    """Calculate the advection term using central finite difference.

    Args:
        omega_left: Vorticity at (i-1, j).
        omega_right: Vorticity at (i+1, j).
        omega_top: Vorticity at (i, j+1).
        omega_down: Vorticity at (i, j-1).
        psi_left: Stream function at (i-1, j).
        psi_right: Stream function at (i+1, j).
        psi_top: Stream function at (i, j+1).
        psi_down: Stream function at (i, j-1).
        h: Grid spacing.

    Returns:
        Advection term value at grid point (i, j).
    """
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
    """Calculate the Laplacian for viscous diffusion using central difference.

    Args:
        omega_left: Vorticity at (i-1, j).
        omega_right: Vorticity at (i+1, j).
        omega_center: Vorticity at (i, j).
        omega_down: Vorticity at (i, j-1).
        omega_top: Vorticity at (i, j+1).
        h: Grid spacing.

    Returns:
        Laplacian of vorticity at grid point (i, j).
    """
    inv_h2 = 1.0 / (h * h)
    # combine both the diffusion terms in the x and y direction together
    laplacian = (
        omega_right + omega_left + omega_top + omega_down - 4.0 * omega_center
    ) * inv_h2
    return laplacian


# -----------------------------------------------------------------------------
# Warp kernels for SSP-RK3 timestepping
# -----------------------------------------------------------------------------


@wp.kernel
def decaying_turbulence_initializer(
    n: int,
    k_cutoff: wp.float32,
    s: wp.int32,
    k_mag: wp.array2d(dtype=wp.float32),
    zeta: wp.array2d(dtype=wp.float32),
    eta: wp.array2d(dtype=wp.float32),
    omega_hat_init: wp.array2d(dtype=wp.vec2f),
):
    """Initialize the vorticity field in Fourier space for decaying turbulence.

    Args:
        n: Size of the simulation domain.
        k_cutoff: Wavenumber magnitude at which maximum of energy spectrum lies.
        s: Shape parameter of the energy spectrum.
        k_mag: Wavenumber magnitude array.
        zeta: First phase function for phase randomization.
        eta: Second phase function for phase randomization.
        omega_hat_init: Output vorticity field in Fourier space.
    """
    i, j = wp.tid()

    amplitude = wp.sqrt(
        (k_mag[i, j] / wp.pi) * energy_spectrum(k_mag[i, j], s, k_cutoff)
    )
    phase = phase_randomizer(n, zeta, eta, i, j)
    omega_hat_init[i, j] = wp.vec2f(
        amplitude * wp.cos(phase), amplitude * wp.sin(phase)
    )


@wp.kernel
def viscous_advection_rk3_kernel(
    n: int,
    h: float,
    re: float,
    dt: float,
    coeff0: float,
    coeff1: float,
    coeff2: float,
    omega_0: wp.array2d(dtype=float),
    omega_1: wp.array2d(dtype=float),
    psi: wp.array2d(dtype=float),
    rhs: wp.array2d(dtype=float),
):
    """Perform a single substep of SSP-RK3.

    Args:
        n: Grid size.
        h: Grid spacing.
        re: Reynolds number.
        dt: Time step size.
        coeff0: SSP-RK3 coefficient for omega_0.
        coeff1: SSP-RK3 coefficient for omega_1.
        coeff2: SSP-RK3 coefficient for RHS.
        omega_0: Vorticity field at the beginning of the time step.
        omega_1: Vorticity field at the end of the time step.
        psi: Stream function field.
        rhs: Temporarily stores diffusion + advection terms.
    """
    i, j = wp.tid()

    # obtain the neighboring indices for the [i, j]th cell in a periodic square box
    left_idx = cyclic_index(i - 1, n)
    right_idx = cyclic_index(i + 1, n)
    top_idx = cyclic_index(j + 1, n)
    down_idx = cyclic_index(j - 1, n)

    # compute viscous diffusion term
    rhs[i, j] = (1.0 / re) * diffusion(
        omega_1[left_idx, j],
        omega_1[right_idx, j],
        omega_1[i, j],
        omega_1[i, down_idx],
        omega_1[i, top_idx],
        h,
    )

    # add advection term
    rhs[i, j] += advection(
        omega_1[left_idx, j],
        omega_1[right_idx, j],
        omega_1[i, top_idx],
        omega_1[i, down_idx],
        psi[left_idx, j],
        psi[right_idx, j],
        psi[i, top_idx],
        psi[i, down_idx],
        h,
    )

    # perform RK update
    omega_1[i, j] = (
        coeff0 * omega_0[i, j] + coeff1 * omega_1[i, j] + coeff2 * dt * rhs[i, j]
    )


# -----------------------------------------------------------------------------
# Helper kernels for Poisson solver and 2-D FFT in the spectral space
# -----------------------------------------------------------------------------
@wp.kernel
def copy_float_to_vec2(
    omega: wp.array2d(dtype=wp.float32), omega_complex: wp.array2d(dtype=wp.vec2f)
):
    """Copy real vorticity to a complex array with zero imaginary part.

    Args:
        omega: Input real-valued vorticity array.
        omega_complex: Output complex array where real part is omega, imaginary is 0.
    """
    i, j = wp.tid()
    omega_complex[i, j] = wp.vec2f(omega[i, j], 0.0)


@wp.kernel
def fft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Perform 1-D FFT on each row using wp.tile_fft().

    Args:
        x: Input complex array of shape (N, N).
        y: Output complex array of shape (N, N) storing FFT results.
    """
    i, _, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_fft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel
def ifft_tiled(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Perform 1-D inverse FFT on each row using wp.tile_ifft().

    Args:
        x: Input complex array of shape (N, N).
        y: Output complex array of shape (N, N) storing IFFT results.
    """
    i, _, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0))
    wp.tile_ifft(a)
    wp.tile_store(y, a, offset=(i * TILE_M, 0))


@wp.kernel
def tiled_transpose(x: wp.array2d(dtype=wp.vec2f), y: wp.array2d(dtype=wp.vec2f)):
    """Transpose a 2-D array.

    Args:
        x: Input complex array.
        y: Output complex array storing the transpose of x.
    """
    i, j = wp.tid()
    t = wp.tile_load(
        x,
        shape=(TILE_TRANSPOSE_DIM, TILE_TRANSPOSE_DIM),
        offset=(i * TILE_TRANSPOSE_DIM, j * TILE_TRANSPOSE_DIM),
        storage="shared",
    )
    t_transposed = wp.tile_transpose(t)
    wp.tile_store(
        y, t_transposed, offset=(j * TILE_TRANSPOSE_DIM, i * TILE_TRANSPOSE_DIM)
    )


@wp.kernel
def multiply_k2_inverse(
    k2i: wp.array2d(dtype=wp.float32),
    omega_hat: wp.array2d(dtype=wp.vec2f),
    psi_hat: wp.array2d(dtype=wp.vec2f),
):
    """Solve Poisson equation in Fourier space.

    Args:
        k2i: Precomputed 1/|k|^2 array.
        omega_hat: Fourier transform of vorticity.
        psi_hat: Output Fourier transform of stream function.
    """
    i, j = wp.tid()
    psi_hat[i, j] = omega_hat[i, j] * k2i[i, j]


@wp.kernel
def extract_real(
    complex_array: wp.array2d(dtype=wp.vec2f), real_array: wp.array2d(dtype=wp.float32)
):
    """Extract real part from a complex array.

    Args:
        complex_array: Input complex array (vec2f where .x is real part).
        real_array: Output real array.
    """
    i, j = wp.tid()
    real_array[i, j] = complex_array[i, j].x


@wp.kernel
def scale_array(scale: wp.float32, arr: wp.array2d(dtype=wp.float32)):
    """Scale array by multiplying with a scale factor.

    Args:
        scale: Scale factor to multiply each element by.
        arr: Array to normalize (modified in-place).
    """
    i, j = wp.tid()
    arr[i, j] = arr[i, j] * scale


class Example:
    """Implement 2-D flow in a periodic box using vorticity-streamfunction formulation."""

    def __init__(
        self, n: int = N_GRID, re: float = RE, dt: float = DT, length: float = LEN
    ) -> None:
        """Initialize the 2-D Navier Stokes solver in a square box.

        Args:
            n: Square grid resolution (must match TILE_N for tile-based FFT in Warp).
            re: Reynolds number.
            dt: Time step size.
            length: Physical domain size (default 2*pi).
        """
        self.n = n
        self.length = length
        self.h = self.length / self.n
        self.re = re
        self.dt = dt

        # define SSP-RK3 coefficients
        self.rk3_coeffs = [
            [1.0, 0.0, 1.0],
            [3.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        ]

        # initialize fields
        self._init_fields()

    def _init_fields(self) -> None:
        """Initialize all the required variables for the simulation."""
        # allocate warp arrays for vorticity, stream-function, and RHS of NS equation
        self.omega_0 = wp.zeros((self.n, self.n), dtype=wp.float32)
        self.omega_1 = wp.zeros((self.n, self.n), dtype=wp.float32)
        self.psi = wp.zeros((self.n, self.n), dtype=wp.float32)
        self.rhs = wp.zeros((self.n, self.n), dtype=wp.float32)

        # precompute 1/k^2 for spectral Poisson solver (avoid division by zero at k=0)
        k = np.fft.fftfreq(self.n, d=1.0 / self.n)
        kx, ky = np.meshgrid(k, k)
        k2 = kx**2 + ky**2
        k2i = np.zeros_like(k2)
        nonzero = k2 != 0
        k2i[nonzero] = 1.0 / k2[nonzero]
        self.k2i = wp.array2d(k2i.astype(np.float32), dtype=wp.float32)

        # allocate temporary warp arrays for spectral Poisson solver
        self.omega_complex = wp.zeros((self.n, self.n), dtype=wp.vec2f)
        self.fft_temp_1 = wp.zeros((self.n, self.n), dtype=wp.vec2f)
        self.fft_temp_2 = wp.zeros((self.n, self.n), dtype=wp.vec2f)

        # compute initial vorticity distribution for decaying turbulence
        k = np.fft.fftfreq(self.n, d=1.0 / self.n)
        k_mag_np = np.sqrt(k**2 + k[:, np.newaxis] ** 2)
        k_mag = wp.array2d(k_mag_np, dtype=wp.float32)

        rng = np.random.default_rng(42)
        zeta_np = 2 * np.pi * rng.random((self.n // 2 + 1, self.n // 2 + 1))
        eta_np = 2 * np.pi * rng.random((self.n // 2 + 1, self.n // 2 + 1))
        zeta = wp.array2d(zeta_np, dtype=wp.float32)
        eta = wp.array2d(eta_np, dtype=wp.float32)

        # set parameters for energy spectrum
        k_cutoff = 12.0
        s = 3

        wp.launch(
            decaying_turbulence_initializer,
            dim=(self.n, self.n),
            inputs=[self.n, k_cutoff, s, k_mag, zeta, eta],
            outputs=[self.omega_complex],
        )

        # compute IFFT of self.omega_complex field
        self._fft_2d(ifft_tiled, self.omega_complex, self.fft_temp_1)

        # extract real part get initial vorticity field
        wp.launch(
            extract_real,
            dim=(self.n, self.n),
            inputs=[self.fft_temp_1],
            outputs=[self.omega_0],
        )

        # for initial distribution, set both omega_1 and omega_0 to be the same
        self.omega_1 = self.omega_0

        print(self.omega_0.numpy().min(), self.omega_0.numpy().max())

        # solve initial Poisson equation to get psi from initial vorticity field
        self._solve_poisson()

    def _fft_2d(
        self,
        fft_kernel: wp.Kernel,
        input_arr: wp.array2d(dtype=wp.vec2f),
        output_arr: wp.array2d(dtype=wp.vec2f),
    ) -> None:
        """Perform 2-D FFT or IFFT using row-wise transform + transpose pattern.

        Args:
            fft_kernel: Either fft_tiled or ifft_tiled.
            input_arr: Input complex array.
            output_arr: Output complex array.
        """
        # perform rowwise FFT/IFFT
        wp.launch_tiled(
            fft_kernel,
            dim=[self.n, 1],
            inputs=[input_arr],
            outputs=[self.fft_temp_1],
            block_dim=BLOCK_DIM,
        )

        wp.launch_tiled(
            tiled_transpose,
            dim=(self.n // TILE_TRANSPOSE_DIM, self.n // TILE_TRANSPOSE_DIM),
            inputs=[self.fft_temp_1],
            outputs=[self.fft_temp_2],
            block_dim=TILE_TRANSPOSE_DIM * TILE_TRANSPOSE_DIM,
        )

        # perform columnwise FFT/IFFT
        wp.launch_tiled(
            fft_kernel,
            dim=[self.n, 1],
            inputs=[self.fft_temp_2],
            outputs=[output_arr],
            block_dim=BLOCK_DIM,
        )

    def _solve_poisson(self) -> None:
        """Solve the Poisson equation using FFT.

        Solve (del^2/del x^2 + del^2/del y^2)(psi) = -omega_1.
        psi_hat(kx, ky) = omega_hat(kx, ky) / ||k||^2 for periodic 2-D domain.
        2-D FFT is computed as sequence of 1-D FFT along rows, transpose, and 1-D FFT along rows.
        """
        # convert updated vorticity at any RK step from wp.float32 to wp.vec2f
        wp.launch(
            copy_float_to_vec2,
            dim=(self.n, self.n),
            inputs=[self.omega_1],
            outputs=[self.omega_complex],
        )

        # perform forward FFT
        self._fft_2d(fft_tiled, self.omega_complex, self.fft_temp_1)

        # multiply by 1/k^2 to solve Poisson in Fourier space
        wp.launch(
            multiply_k2_inverse,
            dim=(self.n, self.n),
            inputs=[self.k2i, self.fft_temp_1],
            outputs=[self.fft_temp_2],
        )

        # perform inverse FFT
        self._fft_2d(ifft_tiled, self.fft_temp_2, self.fft_temp_1)

        # extract real part and normalize
        wp.launch(
            extract_real,
            dim=(self.n, self.n),
            inputs=[self.fft_temp_1],
            outputs=[self.psi],
        )
        wp.launch(
            scale_array,
            dim=(self.n, self.n),
            inputs=[wp.float32(1.0 / (self.n * self.n)), self.psi],
        )

    def step(self) -> None:
        """Advance simulation by one timestep using SSP-RK3."""
        for stage_coeff in self.rk3_coeffs:
            # c0, c1, c2 = self.rk3_coeffs[stage]

            # zero the RHS array
            self.rhs.zero_()

            # compute RHS and update omega_1
            wp.launch(
                viscous_advection_rk3_kernel,
                dim=(self.n, self.n),
                inputs=[
                    self.n,
                    self.h,
                    self.re,
                    self.dt,
                    stage_coeff[0],
                    stage_coeff[1],
                    stage_coeff[2],
                    self.omega_0,
                    self.omega_1,
                    self.psi,
                    self.rhs,
                ],
            )

            # update streamfunction from new vorticity (in omega_1)
            self._solve_poisson()

        # copy omega_1 to omega_0 for next timestep
        wp.copy(self.omega_0, self.omega_1)

    def step_and_render_frame(
        self, frame_num: int, img: Optional[Any] = None
    ) -> tuple[Any, ...]:
        """Advance simulation by one timestep and update the matplotlib image.

        Args:
            frame_num: Current frame number (required by FuncAnimation).
            img: Matplotlib image object to update.

        Returns:
            Tuple containing the updated image for blitting.
        """

        # replay graph for all remaining frames
        wp.capture_launch(self.step_graph)

        if img:
            img.set_array(self.omega_1.numpy().T)

        return (img,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="2-D Turbulence Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the default Warp device.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=20000,
        help="Total number of frames.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()

        # capture first step in a CUDA graph
        with wp.ScopedCapture() as capture:
            example.step()
        example.step_graph = capture.graph

        if args.headless:
            # replay graph for all remaining frames
            for _ in range(args.num_frames):
                wp.capture_launch(example.step_graph)
        else:
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            fig = plt.figure()

            img = plt.imshow(
                example.omega_1.numpy().T,
                origin="lower",
                cmap="twilight",
                animated=True,
                interpolation="antialiased",
                vmin=-15,
                vmax=15,
            )

            seq = anim.FuncAnimation(
                fig,
                example.step_and_render_frame,
                fargs=(img,),
                frames=args.num_frames,
                blit=True,
                interval=1,
                repeat=False,
            )

            plt.show()
