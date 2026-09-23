# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT

"""Validation shared by the completed MuJoCo Warp demo and scaling study."""

import mujoco_warp as mjw
import numpy as np
import warp as wp


def require_cuda():
    """Select the current CUDA device without silently timing a CPU fallback."""
    wp.init()
    device = wp.get_device()
    if not device.is_cuda:
        raise RuntimeError("This MuJoCo Warp example requires an NVIDIA CUDA device.")
    return device


def check_mjwarp_state(data) -> list[str]:
    """Reject invalid state or lost physics, returning solver-limit diagnostics.

    MuJoCo Warp 3.12 accumulates Data.overflow flags until reset_data. Checking
    after timing therefore detects capacity loss in warmup or timed steps
    without adding host synchronization to the measured loop. Iteration limits
    are convergence diagnostics, distinct from truncated contacts/constraints.
    """
    flags = int(np.bitwise_or.reduce(data.overflow.numpy(), initial=0))
    iteration_mask = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
    capacity_flags = flags & ~iteration_mask
    if capacity_flags:
        names = [flag.name for flag in mjw.OverflowType if capacity_flags & int(flag)]
        raise RuntimeError(
            "MuJoCo Warp capacity overflow: " + ", ".join(names)
            + ". Increase the indicated contact/constraint/collision budgets; discard this run."
        )
    for name in ("qpos", "qvel", "qacc"):
        if not np.isfinite(getattr(data, name).numpy()).all():
            raise RuntimeError(f"MuJoCo Warp produced non-finite {name}; discard this run.")
    return [flag.name for flag in mjw.OverflowType if flags & int(flag)]
