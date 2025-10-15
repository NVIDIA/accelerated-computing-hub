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

module sm
contains
function wtime() result(t)
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=REAL64) ::  t
  integer(kind=INT64) :: c, r
  call system_clock(count = c, count_rate = r)
  t = real(c,REAL64) / real(r,REAL64)
end function wtime
end module

program main
  use, intrinsic :: iso_fortran_env
  use sm
  ! DONE: import cutensorex module
  use cutensorex
  implicit none
  integer :: ni, nj, nk, niter
  real(8), allocatable, dimension(:,:) :: a, b, d
  integer :: i, j, k, it
  real(kind=REAL64) :: t1, t2, flops
  character(100) :: args

  call get_command_argument(1,args)
  read(args,*) ni
  call get_command_argument(2,args)
  read(args,*) nk
  call get_command_argument(3,args)
  read(args,*) nj
  call get_command_argument(4,args)
  read(args,*) niter
          
  allocate(a(ni, nk), b(nk, nj), d(ni, nj))
  call random_number(a)
  call random_number(b)

  ! warmup:
  ! DONE: implement using the matmul intrinsic
  d = matmul(a,b)
    
  ! time:
  print *,"gemm dims ni=",ni," nk=",nk," nj=",nj
  t1 = wtime()
  do it = 1, niter
    ! DONE: implement using the matmul intrinsic
    d = d + matmul(a,b)
  end do
  t2 = wtime()

  flops = 2. * ni * nj * nk * niter / (t2 - t1) * 1e-9
  print *,"GFLOP/s=",flops
end program