#! /bin/bash
#
# Run checks for the Warp tutorial.
#
# Usage:
#   ./test.bash                    # GPU availability and simulation checks
#   ./test.bash -k cpu             # forward filters to pytest

set -euo pipefail

TUTORIAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
nvidia-smi
python -m pytest "$TUTORIAL_DIR/test" "$@"
