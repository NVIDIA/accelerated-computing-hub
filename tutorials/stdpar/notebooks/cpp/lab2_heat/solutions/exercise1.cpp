/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//! Solves heat equation in 2D, see the README.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <execution>
#include <fstream>
#include <iostream>
#include <mdspan>
#include <mpi.h>
#include <numeric>
#include <ranges>
#include <vector>

using grid_t = std::mdspan<double, std::dextents<std::size_t, 2>, std::layout_right>;

// Problem parameters
struct parameters {
  double dx, dt;
  long nx, ny, ni;
  int rank = 0, nranks = 1;

  static constexpr double alpha() { return 1.0; } // Thermal diffusivity

  parameters(int argc, char *argv[]);

  long nit() { return ni; }
  long nout() { return ni / 10; }
  long nx_global() { return nx * nranks; }
  long ny_global() { return ny; }
  double gamma() { return alpha() * dt / (dx * dx); }
  long n() { return ny * (nx + 2 /* 2 halo layers */); }
};

double stencil(grid_t u_new, grid_t u_old, long x, long y, parameters p);

// 2D grid of indicies
struct grid {
  long x_begin, x_end, y_begin, y_end;
};

double apply_stencil(grid_t u_new, grid_t u_old, grid g, parameters p) {
  // DONE Create one iota range per dimension for [g.x_begin,g.x_end) and [g.y_begin,g.y_end).
  auto xs = std::views::iota(g.x_begin, g.x_end);
  auto ys = std::views::iota(g.y_begin, g.y_end);
  // DONE: Construct a cartesian_product range from the two iota ranges: [g.x_begin,g.x_end)x[g.y_begin,g.y_end).
  auto ids = std::views::cartesian_product(xs, ys);
  // DONE: Use the std::transform_reduce algorithm to apply the stencil in parallel to each element and sum the energies:
  return std::transform_reduce(
    // DONE: Use the std::execution::par parallel execution policy
    std::execution::par,
    // DONE: iterate over the cartesian_product range
    ids.begin(), ids.end(),
    // DONE: initialize the energy to zero
    0.,
    // DONE: use std::plus to sum the energies
    std::plus{},
    // DONE: Use a lambda that applies the stencil to one element and returns its energy:
    [u_new, u_old, p](auto idx) {
      // DONE [within lambda]: Extract the 1D indices from the tuple of indices:
      auto [x, y] = idx;
      // DONE [within lambda]: Apply the stencil and return the energy.
      return stencil(u_new, u_old, x, y, p);
  });
}

// Initial condition
void initial_condition(grid_t u_new, grid_t u_old) {
  // DONE: parallelize using the std::fill_n parallel algorithm
  std::fill_n(std::execution::par, u_old.data_handle(), u_old.size(), 0.0);
  std::fill_n(std::execution::par, u_new.data_handle(), u_new.size(), 0.0);
}

// These evolve the solution of different parts of the local domain.
double inner(grid_t u_new, grid_t u_old, parameters p);
double prev (grid_t u_new, grid_t u_old, parameters p); 
double next (grid_t u_new, grid_t u_old, parameters p);

int main(int argc, char *argv[]) {
  // Parse CLI parameters
  parameters p(argc, argv);

  // Initialize MPI with multi-threading support
  int mt;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mt);
  if (mt != MPI_THREAD_MULTIPLE) {
    std::cerr << "MPI cannot be called from multiple host threads" << std::endl;
    std::terminate();
  }
  MPI_Comm_size(MPI_COMM_WORLD, &p.nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &p.rank);

#if defined(_NVHPC_STDPAR_GPU)
  int dev = 0;
  cudaGetDevice(&dev);

  char bus_id[64];
  cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), dev);

  // We use `printf` to serialize the output.
  printf("rank %d using GPU %s\n", p.rank, bus_id);
#endif

  // Allocate memory
  std::vector<double> u_new_data(p.n()), u_old_data(p.n());
  grid_t u_new{u_new_data.data(), p.nx+2, p.ny};
  grid_t u_old{u_old_data.data(), p.nx+2, p.ny};

  // Initial condition
  initial_condition(u_new, u_old);

  // Time loop
  using clk_t = std::chrono::steady_clock;
  auto start = clk_t::now();

  for (long it = 0; it < p.nit(); ++it) {
    // Evolve the solution:
    double energy = prev(u_new, u_old, p) + next(u_new, u_old, p) + inner(u_new, u_old, p);

    // Reduce the energy across all neighbors to the rank == 0, and print it if necessary:
    MPI_Reduce(p.rank == 0 ? MPI_IN_PLACE : &energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (p.rank == 0 && it % p.nout() == 0) {
      std::cerr << "E(t=" << it * p.dt << ") = " << energy << std::endl;
    }
    std::swap(u_new, u_old);
  }

  auto time = std::chrono::duration<double>(clk_t::now() - start).count();
  auto grid_size = static_cast<double>(p.nx * p.ny * sizeof(double) * 2) * 1e-9; // GB
  auto memory_bw = grid_size * static_cast<double>(p.nit()) / time;             // GB/s
  if (p.rank == 0) {
    std::cerr << "Rank " << p.rank << ": local domain " << p.nx << "x" << p.ny << " (" << grid_size << " GB): " 
              << memory_bw << " GB/s " << time << " s" << std::endl;
    std::cerr << "All ranks: global domain " << p.nx_global() << "x" << p.ny_global() << " (" << (grid_size * p.nranks) << " GB): "
              << memory_bw * p.nranks << " GB/s " << time << " s" << std::endl;
  }

  // Write output to file
  MPI_File f;
  MPI_File_open(MPI_COMM_WORLD, "output", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);
  auto header_bytes = 2 * sizeof(long) + sizeof(double);
  auto values_per_rank = p.nx * p.ny;
  auto values_bytes_per_rank = values_per_rank * sizeof(double);
  MPI_File_set_size(f, header_bytes + values_bytes_per_rank * p.nranks);
  MPI_Request req[3] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  if (p.rank == 0) {
    long total[2] = {p.nx * p.nranks, p.ny};
    double time = p.nit() * p.dt;
    MPI_File_iwrite_at(f, 0, total, 2, MPI_UINT64_T, &req[1]);
    MPI_File_iwrite_at(f, 2 * sizeof(long), &time, 1, MPI_DOUBLE, &req[2]);
  }
  auto values_offset = header_bytes + p.rank * values_bytes_per_rank;
  auto u_out_data = std::vector<double>(p.n());
  using grid_io_t = std::mdspan<double, std::dextents<std::size_t, 2>, std::layout_right>;
  grid_io_t u_out{u_out_data.data(), p.nx+2, p.ny};
  auto is = std::views::iota(0, (int)u_out.extent(0));
  auto js = std::views::iota(0, (int)u_out.extent(1));
  auto ids = std::views::cartesian_product(is, js);
  std::for_each(std::execution::par, ids.begin(), ids.end(), [u_out, u_old](auto idx) {
     auto [i, j] = idx;
     u_out(i, j) = u_old(i, j);
  });
  MPI_File_iwrite_at(f, values_offset, u_out.data_handle() + p.ny, values_per_rank, MPI_DOUBLE, &req[0]);
  MPI_Waitall(p.rank == 0 ? 3 : 1, req, MPI_STATUSES_IGNORE);
  MPI_File_close(&f);

  MPI_Finalize();
  return 0;
}

// Reads command line arguments to initialize problem size
parameters::parameters(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "ERROR: incorrect arguments" << std::endl;
    std::cerr << "  " << argv[0] << " <nx> <ny> <ni>" << std::endl;
    std::terminate();
  }
  nx = std::stoll(argv[1]);
  ny = std::stoll(argv[2]);
  ni = std::stoll(argv[3]);
  dx = 1.0 / nx;
  dt = dx * dx / (5. * alpha());
}

// Finite-difference stencil
double stencil(grid_t u_new, grid_t u_old, long x, long y, parameters p) {
  if (y == 1) u_old(x, y-1) = 0;
  if (y == (p.ny - 2)) u_old(x, y+1) = 0;

  // These boundary conditions are only imposed by the ranks at the end of the domain:
  if (p.rank == 0 && x == 1) u_old(x-1, y) = 1;
  if (p.rank == (p.nranks - 1) && x == p.nx) u_old(x+1, y) = 0;

  u_new(x, y) = (1. - 4. * p.gamma()) * u_old(x, y) + p.gamma() * (u_old(x+1, y) + u_old(x-1, y) +
                                                                   u_old(x, y+1) + u_old(x, y-1));

  return u_new(x, y) * p.dx * p.dx;
}

// Evolve the solution of the interior part of the domain
// which does not depend on data from neighboring ranks
double inner(grid_t u_new, grid_t u_old, parameters p) {
  grid g{.x_begin = 2, .x_end = p.nx, .y_begin = 1, .y_end = p.ny - 1};
  return apply_stencil(u_new, u_old, g, p);
}

// Evolve the solution of the part of the domain that 
// depends on data from the previous MPI rank (rank - 1)
double prev(grid_t u_new, grid_t u_old, parameters p) {
  thread_local std::vector<double> halos_tx((std::size_t)p.ny);
  thread_local std::vector<double> halos_rx((std::size_t)p.ny);
  // Send window cells, receive halo cells
  if (p.rank > 0) {
    // Copy halos to transmit into the transmit buffer
    std::for_each_n(std::execution::par, std::views::iota(0).begin(), p.ny, [halos_tx = halos_tx.data(), u_old](int i) {
       halos_tx[i] = u_old(1, i); 
    });
    // Send bottom boundary to bottom rank and receive top boundary from bottom rank
    MPI_Sendrecv(halos_tx.data(), p.ny, MPI_DOUBLE, p.rank - 1, 0, 
                 halos_rx.data(), p.ny, MPI_DOUBLE, p.rank - 1, 0, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Copy data from the receive buffer into the grid
    std::for_each_n(std::execution::par, std::views::iota(0).begin(), p.ny, [halos_rx = halos_rx.data(), u_old](int i) {
       u_old(0, i) = halos_rx[i]; 
    });
  }
  // Compute prev boundary
  grid g{.x_begin = 1, .x_end = 2, .y_begin= 1, .y_end = p.ny - 1};
  return apply_stencil(u_new, u_old, g, p);
}

// Evolve the solution of the part of the domain that 
// depends on data from the next MPI rank (rank + 1)
double next(grid_t u_new, grid_t u_old, parameters p) {
  // Allocate data for transmitting and receiving halos:
  thread_local std::vector<double> halos_tx((std::size_t)p.ny);
  thread_local std::vector<double> halos_rx((std::size_t)p.ny);
    
  if (p.rank < p.nranks - 1) {
    // Copy halos to transmit into the transmit buffer
    std::for_each_n(std::execution::par, std::views::iota(0).begin(), p.ny, [halos_tx = halos_tx.data(), u_old, p](int i) {
      halos_tx[i] = u_old(p.nx, i); 
    });
    // Receive bottom boundary from top rank and send top boundary to top rank
    MPI_Sendrecv(halos_tx.data(), p.ny, MPI_DOUBLE, p.rank + 1, 0, 
                 halos_rx.data(), p.ny, MPI_DOUBLE, p.rank + 1, 0, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Copy received halos to the u_old solution buffer
    std::for_each_n(std::execution::par, std::views::iota(0).begin(), p.ny, [halos_rx = halos_rx.data(), u_old, p](int i) {
      u_old(p.nx+1, i) = halos_rx[i]; 
    });
  }
  // Compute next boundary
  grid g{.x_begin = p.nx, .x_end = p.nx + 1, .y_begin = 1, .y_end = p.ny - 1};
  return apply_stencil(u_new, u_old, g, p);
}
