#! /bin/bash
#
# Entrypoint for the jupyter service. Runs as root, then switches to user.

set -euo pipefail

# Switch to user and run user-level entrypoint
exec gosu "${ACH_TARGET_USER}" /accelerated-computing-hub/brev/entrypoint-jupyter-user.bash "$@"
