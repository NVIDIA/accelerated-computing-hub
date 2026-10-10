#! /bin/bash
#
# Run checks for the Sim2Real Blogs tutorial.
#
# Usage:
#   bash test.bash                    # GPU availability and simulation checks
#   bash test.bash -k cpu             # forward filters to pytest

set -euo pipefail

TUTORIAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
nvidia-smi
# Bound the complete correctness suite; this never enables benchmark measurements.
# GNU timeout terminates its process group on expiry; Docker owns container cleanup.
timeout --signal=TERM --kill-after=15s 1800 python -m pytest "$TUTORIAL_DIR/test" "$@"
