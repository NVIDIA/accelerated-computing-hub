# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation
# SPDX-License-Identifier: MIT
"""Compatibility import for the shared CPU/GPU task implementation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim2real_shared import pick_place_common as _implementation

# Preserve the module identity, including mutable robot-layout globals.
sys.modules[__name__] = _implementation
