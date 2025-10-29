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

! DAXPY: Y + Y + A * X  and sum(Y)
subroutine daxpy_sum(x, y, n, a, s)
  use, intrinsic :: iso_fortran_env
  implicit none
  real(kind=8), dimension(:) :: x, y
  real(kind=8) :: a, s
  integer :: n, i  
  ! TODO: parallelize using do-concurrent
  s = 0.
  do i = 1, n
    y(i) = y(i) + a * x(i)
    s = s + y(i)
  end do   
end subroutine

! Get wall clock time
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
  implicit none
  real(kind=8), dimension(:), allocatable :: x, y
  real(kind=8) :: a = 2.0, s = 0.0, s_should = 0.0
  integer :: n, i, niter
  real(kind=REAL64) :: t0, t1, c
  character(100) :: args

  call get_command_argument(1,args)
  read(args,*) n
  call get_command_argument(2,args)
  read(args,*) niter

  allocate(x(n), y(n))

  ! Intialize vectors `x` and `y`
  do concurrent (i = 1:n) default(none) shared(x, y)
    x(i)  = i
    y(i)  = 2.
  end do

  ! check solution
  call daxpy_sum(x, y, n, a, s)
  do i = 1, n
    if (abs(y(i) - (a * i + 2.)) .ge. 1.e-4) then
      print *, "ERROR!",i,x(i),y(i),(a * i + 2.)
      return
    endif
    s_should = s_should + (a * i + 2.)
  end do
  if (abs(s - s_should) .ge. 1.e-4) then
    print *, "ERROR!",s,s_should
    return
  endif
  print *, "OK!"

  ! benchmark
  call daxpy_sum(x, y, n, a, s) ! warmup
  t0 = wtime()
  do i = 1, niter
    call daxpy_sum(x, y, n, a, s)
  end do
  t1 = wtime()
    
  c = (3. * n * 8. * niter) / (t1 - t0) * 1e-9
  print *, (2*n*8*1e-9), ' GB, ', c, ' GB/s'
end program