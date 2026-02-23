#! /bin/bash
#
# User-level entrypoint for the jupyter service. Sets up Jupyter and runs it.

set -euo pipefail

export HOME="${ACH_TARGET_HOME}"

# Generate Jupyter plugin settings
/accelerated-computing-hub/brev/jupyter-generate-plugin-settings.bash

mkdir -p /accelerated-computing-hub/logs

# Forward ports from other Docker Compose services to localhost so that
# jupyter-server-proxy can reach them (it only proxies to localhost).
if command -v socat &> /dev/null; then
  for FORWARD in ${ACH_JUPYTER_PORT_FORWARDS:-}; do
    LOCAL_PORT="${FORWARD%%:*}"
    REMOTE="${FORWARD#*:}"
    socat "TCP-LISTEN:${LOCAL_PORT},fork,reuseaddr" "TCP:${REMOTE}" &
  done
fi

# Set the preferred directory to the current working directory, which is set by Docker Compose.
ARGS="--ServerApp.preferred_dir=${PWD:-/}"

if [ -n "${1:-}" ]; then
  ARGS="--LabApp.default_url=${1}"
fi

exec python -m jupyter lab ${ARGS}
