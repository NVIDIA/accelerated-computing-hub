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
python -m pytest "$TUTORIAL_DIR/test" "$@"
