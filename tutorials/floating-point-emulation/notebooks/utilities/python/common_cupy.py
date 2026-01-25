# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import cupy


def time_cupy(fun, ncycles, *args):
    start, stop = cupy.cuda.Event(), cupy.cuda.Event()
    out = fun(*args)

    start.record(None)
    for _ in range(ncycles):
        out = fun(*args)  # noqa: F841
    stop.record(None)
    stop.synchronize()

    time_ms = cupy.cuda.get_elapsed_time(start, stop) / ncycles

    return time_ms