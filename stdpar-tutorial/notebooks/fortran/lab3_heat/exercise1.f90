!
! SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
! SPDX-License-Identifier: MIT
!
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.
!

module heat
 use, intrinsic :: iso_fortran_env
 use mpi
 implicit none
 
 ! Simulation parameters
 type param
  real(kind=8) :: dx, dt
  integer :: nx, ny, ni, rank, nranks
 end type
 
 ! Slice of simulation grid (x_start, x_end) x (y_start, y_end)
 type grid
  integer :: x_start, x_end, y_start, y_end
 end type

contains

 ! Get wall clock time
 function wtime() result(t)
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=REAL64) ::  t
  integer(kind=INT64) :: c, r
  call system_clock(count = c, count_rate = r)
  t = real(c,REAL64) / real(r,REAL64)
 end function wtime

 ! Parse command line arguments
 subroutine parse_cli( p )
  type(param), intent(out) :: p
  character(100) :: args
  call get_command_argument(1,args)
  read(args,*) p%nx
  call get_command_argument(2,args)
  read(args,*) p%ny
  call get_command_argument(3,args)
  read(args,*) p%ni
  p%dx = 1.0 / p%nx
  p%dt = p%dx * p%dx / 5.
 end subroutine

 ! Computes the gamma factor: alpha * dt / (dx^2)
 ! TODO: make gamma pure
 function gamma(p) result(r)
  use, intrinsic :: iso_fortran_env
  implicit none
  type(param), intent(in) :: p
  real(kind=REAL64) ::  r
  r = p%dt / (p%dx * p%dx)
 end function gamma

 ! Computes the heat-equation stencil and energy of a single element (x, y)
 ! TODO: make stencil pure
 function stencil(u_old, x, y, p) result(o)
  ! TODO: make acc routine seq
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=8), dimension(:,:), intent(in) :: u_old
  integer, intent(in) :: x, y
  type(param), intent(in) :: p
  real(kind=8) :: g, o
  g = gamma(p)
  o = (1. - 4. * g) * u_old(x, y) + g * (u_old(x + 1, y) + u_old(x - 1, y) + u_old(x, y + 1) + u_old(x, y - 1))
 end function

 ! Applies a stencil to a sub-grid of the domain
 subroutine apply_stencil( energy, u_new, u_old, g, p )
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=8), intent(out) :: energy
  real(kind=8), dimension(:,:), intent(out) :: u_new
  real(kind=8), dimension(:,:), intent(inout) :: u_old
  type(grid), intent(in) :: g
  type(param), intent(in) :: p
  integer :: x, y
  energy = 0
  ! TODO: parallelize with do-concurrent and reduce locality-specifier
  do y = g%y_start, g%y_end
    do x = g%x_start, g%x_end
      ! Boundary conditions
      if (x == 2) then
        u_old(x - 1, y) = 1
      end if
      if (x == (p%nx - 1)) then
        u_old(x + 1, y) = 0
      end if
      if (p%rank == 0 .and. y == 2) then
        u_old(x, y - 1) = 0
      end if
      if (p%rank == (p%nranks - 1) .and. y == (p%ny + 1)) then
        u_old(x, y + 1) = 0
      end if
      u_new(x, y) = stencil(u_old, x, y, p)
      energy = energy + u_new(x, y) * p%dx * p%dx
    end do
   end do
 end subroutine

 ! Applies stencil to inner sub-grid of the domain (does not depend on neighbor data)
 subroutine inner(energy, u_new, u_old, p)
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=8), intent(out) :: energy
  real(kind=8), dimension(:,:), intent(out) :: u_new
  real(kind=8), dimension(:,:), intent(inout) :: u_old
  type(param), intent(in) :: p
  type(grid) :: g

  ! Internal domain (2:nx-1,3:ny)
  g%x_start = 2
  g%x_end = p%nx - 1
  g%y_start = 3
  g%y_end = p%ny
  call apply_stencil(energy, u_new, u_old, g, p)
 end subroutine

 ! Applies stencil to "left" sub-grid of the domain (depends on data from MPI rank - 1)
 subroutine prev(energy, u_new, u_old, p)
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=8), intent(out) :: energy
  real(kind=8), dimension(:,:), intent(out) :: u_new
  real(kind=8), dimension(:,:), intent(inout) :: u_old
  type(param), intent(in) :: p
  type(grid)  :: g
  integer :: ierr
  if (p%rank > 0) then
    ! Send left boundary to left rank
    call mpi_send(u_old(1, 2), p%nx, MPI_DOUBLE, p%rank - 1, 0, MPI_COMM_WORLD, ierr)
    ! Receive left halo boundary from left rank
    call mpi_recv(u_old(1, 1), p%nx, MPI_DOUBLE, p%rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
  end if

  ! Left domain (2:nx-1,2:2)
  g%x_start = 2
  g%x_end = p%nx - 1
  g%y_start = 2
  g%y_end = 2
  call apply_stencil(energy, u_new, u_old, g, p)
 end subroutine

 subroutine next(energy, u_new, u_old, p)
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=8), intent(out) :: energy
  real(kind=8), dimension(:,:), intent(out) :: u_new
  real(kind=8), dimension(:,:), intent(inout) :: u_old
  type(param), intent(in) :: p
  type(grid)  :: g
  integer :: ierr
  if (p%rank < (p%nranks - 1)) then
    ! Receive right halo boundary from right rank
    call mpi_recv(u_old(1, p%ny+2), p%nx, MPI_DOUBLE, p%rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
    ! Send top boundary to top rank
    call mpi_send(u_old(1, p%ny+1), p%nx, MPI_DOUBLE, p%rank + 1, 1, MPI_COMM_WORLD, ierr)
  end if

  ! Right domain (2:nx-1,ny+1:ny+1)
  g%x_start = 2
  g%x_end = p%nx - 1
  g%y_start = p%ny + 1
  g%y_end = p%ny + 1
  call apply_stencil(energy, u_new, u_old, g, p)
 end subroutine

 ! Applies the initial condition
 subroutine initial_condition(u_new, u_old, p)
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=8), dimension(:,:), intent(out) :: u_new, u_old
  type(param), intent(in) :: p
  integer :: x, y

  ! TODO: parallelize with do-concurrent
  do x = 1,p%nx
    do y = 1,p%ny
      u_old(x, y) = 0.
      u_new(x, y) = 0.
    end do
  end do
 end subroutine
end module

program main
 use, intrinsic :: iso_fortran_env
 use heat
 use mpi
 implicit none

 real(kind=8), dimension(:,:), allocatable, target :: u_old_, u_new_
 type(param) :: p
 integer :: mt, it, ierr, file, req(3)
 real(kind=8) :: energy_inner, energy_prev, energy_next, energy, t, t1, t2
 integer(kind=8) :: out_sz(2)
 integer(kind=MPI_OFFSET_KIND) :: offset, bytes_per_rank, header_bytes
 real(kind=8), dimension(:,:), pointer :: u_old, u_new, u_tmp

 call parse_cli(p)
 call mpi_init_thread(MPI_THREAD_MULTIPLE, mt, ierr)
 if (mt /= MPI_THREAD_MULTIPLE) then
    print *, "ERROR: MPI failed to initialize"
    return
 end if

 call mpi_comm_size(MPI_COMM_WORLD, p%nranks, ierr)
 call mpi_comm_rank(MPI_COMM_WORLD, p%rank, ierr)

 allocate(u_old_(p%nx, p%ny + 2), u_new_(p%nx, p%ny + 2))  ! Two halo columns
 u_old => u_old_
 u_new => u_new_
 
 call initial_condition(u_new, u_old, p)

 t1 = wtime()
 do it = 1, p%ni-1
    call prev(energy_prev, u_new, u_old, p)
    call next(energy_next, u_new, u_old, p)
    call inner(energy_inner, u_new, u_old, p)
    energy = energy_prev + energy_next + energy_inner
    call MPI_Reduce(energy, energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (p%rank == 0 .and. modulo(it, 1000) == 0) then
        print *, "E(t=", (it * p%dt), ") = ", energy
    end if
    ! Swap the matrices for the next iteration
    u_tmp => u_new
    u_new => u_old
    u_old => u_tmp
 end do
 t2 = wtime()
 
 if (p%rank == 0) then
   print *,"Per rank: GB=",(p%nx * p%ny * 2 * 8 * 1e-9),",GB/s=",((p%ni - 1.) * p%nx * p%ny * 2 * 8 * 1e-9)/(t2 - t1)
   print *,"All rank: GB=",(p%nranks * p%nx * p%ny * 2 * 8 * 1e-9),",GB/s=",((p%ni - 1.) * p%nranks * p%nx * p%ny * 2 * 8 * 1e-9)/(t2 - t1)
 end if

 call mpi_file_open(MPI_COMM_WORLD, "output", IOR(MPI_MODE_CREATE, MPI_MODE_WRONLY), MPI_INFO_NULL, file, ierr)
 header_bytes = (2 + 1) * 8
 bytes_per_rank = p%nx
 bytes_per_rank = (bytes_per_rank * p%ny) * 8
 offset = header_bytes + bytes_per_rank * p%nranks
 call mpi_file_set_size(file, offset, ierr)
 req(1) = MPI_REQUEST_NULL
 req(2) = MPI_REQUEST_NULL
 req(3) = MPI_REQUEST_NULL
 if (p%rank == 0) then
   out_sz(1) = p%nx
   out_sz(2) = p%ny * p%nranks
   t = p%ni * p%dt
   offset = 0
   call mpi_file_iwrite_at(file, offset, out_sz(:), 2, MPI_UINT64_T, req(2), ierr)
   offset = 2 * 8
   call mpi_file_iwrite_at(file, offset, t, 1, MPI_DOUBLE, req(3), ierr)
 end if
 offset = header_bytes + bytes_per_rank * p%rank
 call mpi_file_iwrite_at(file, offset, u_new(1, 2), (p%nx * p%ny), MPI_DOUBLE, req(1), ierr)
 if (p%rank == 0) then
   call mpi_waitall(3, req(:), MPI_STATUSES_IGNORE, ierr)
 else
   call mpi_waitall(1, req(:), MPI_STATUSES_IGNORE, ierr)
 endif
 call mpi_file_close(file, ierr)

 call mpi_finalize(ierr)
end program
