#! /bin/bash
#
# Entrypoint for the jupyter service. Runs as root, then switches to user.

set -euo pipefail

if [ "$(id -u)" = "0" ]; then
    exec gosu "${ACH_TARGET_USER}" /accelerated-computing-hub/brev/entrypoint-jupyter-user.bash "$@"
fi

exec /accelerated-computing-hub/brev/entrypoint-jupyter-user.bash "$@"
