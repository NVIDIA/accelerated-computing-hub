#! /bin/bash
#
# Entrypoint for interactive shell sessions.

set -euo pipefail

# Switch to user and run login shell
exec gosu "${ACH_TARGET_USER}" bash -l
