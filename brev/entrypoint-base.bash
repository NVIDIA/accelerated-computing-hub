#! /bin/bash
#
# Entrypoint for the base service. Runs as root, then switches to user.

set -euo pipefail

if [ "$(id -u)" = "0" ] && [ "${ACH_TARGET_USER}" != "$(id -un)" ]; then
    exec gosu "${ACH_TARGET_USER}" /accelerated-computing-hub/brev/entrypoint-base-user.bash "$@"
else
    exec /accelerated-computing-hub/brev/entrypoint-base-user.bash "$@"
fi
