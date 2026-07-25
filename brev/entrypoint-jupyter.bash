#! /bin/bash
#
# Entrypoint for the jupyter service. Runs as root, then switches to user.

set -euo pipefail

# Keep this wrapper alive so an unrequested Jupyter exit can restart the
# sibling services before the container restart policy starts Jupyter again.
# An operator stopping the container sends TERM/INT to this wrapper; that path
# deliberately does not restart siblings so `compose down` can finish.
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

if [ "$(id -u)" = "0" ] && [ "${ACH_TARGET_USER}" != "$(id -un)" ]; then
    gosu "${ACH_TARGET_USER}" /accelerated-computing-hub/brev/entrypoint-jupyter-user.bash "$@" &
else
    /accelerated-computing-hub/brev/entrypoint-jupyter-user.bash "$@" &
fi
JUPYTER_PID=$!

set +e
wait "${JUPYTER_PID}"
JUPYTER_STATUS=$?
if [ "${TERMINATING}" -eq 1 ] && kill -0 "${JUPYTER_PID}" 2>/dev/null; then
    wait "${JUPYTER_PID}"
    JUPYTER_STATUS=$?
fi
set -e

if [ "${TERMINATING}" -eq 0 ] && \
   [ "${ACH_RESTART_COMPOSE_SERVICES:-}" = "1" ] && \
   [ -S /var/run/docker.sock ]; then
    echo "Jupyter exited with status ${JUPYTER_STATUS}; restarting sibling services."
    if ! python3 /accelerated-computing-hub/brev/restart-compose-services.py; then
        echo "Error: Failed to restart one or more sibling services." >&2
    fi
fi

exit "${JUPYTER_STATUS}"
