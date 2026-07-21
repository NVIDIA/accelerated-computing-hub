#! /bin/bash
#
# Entrypoint for the jupyter service. Runs as root, then switches to user.

set -euo pipefail

# Keep this root wrapper alive so a clean Jupyter shutdown can restart its
# sibling Nsight services through the Docker socket. External container stop
# signals skip the sibling restarts.
TERMINATING=0
JUPYTER_PID=""

# shellcheck disable=SC2317 # Invoked indirectly by the signal traps below.
terminate_jupyter() {
    TERMINATING=1
    if [ -n "${JUPYTER_PID}" ] && kill -0 "${JUPYTER_PID}" 2>/dev/null; then
        kill -TERM "${JUPYTER_PID}"
    fi
}

trap terminate_jupyter TERM INT

gosu "${ACH_TARGET_USER}" /accelerated-computing-hub/brev/entrypoint-jupyter-user.bash "$@" &
JUPYTER_PID=$!

set +e
wait "${JUPYTER_PID}"
JUPYTER_STATUS=$?
if [ "${TERMINATING}" -eq 1 ] && kill -0 "${JUPYTER_PID}" 2>/dev/null; then
    wait "${JUPYTER_PID}"
    JUPYTER_STATUS=$?
fi
set -e

if [ "${TERMINATING}" -eq 0 ] && [ "${JUPYTER_STATUS}" -eq 0 ] && [ -n "${ACH_PORT_FORWARDS:-}" ]; then
    echo "Jupyter exited cleanly; restarting Nsight services."
    if ! python3 /accelerated-computing-hub/brev/restart-compose-services.py nsys ncu; then
        echo "Error: Failed to restart one or more Nsight services." >&2
    fi
fi

exit "${JUPYTER_STATUS}"
