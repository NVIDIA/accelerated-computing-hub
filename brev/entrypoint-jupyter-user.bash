#! /bin/bash
#
# User-level entrypoint for the jupyter service. Sets up Jupyter and runs it.

set -euo pipefail

export HOME="${ACH_TARGET_HOME}"

if [ -n "${ACH_PORT_FORWARDS:-}" ] && ! command -v socat &> /dev/null; then
  echo "Error: ACH_PORT_FORWARDS is configured, but socat is not installed." >&2
  echo "Install socat in the tutorial image to enable Jupyter service proxies." >&2
  exit 1
fi

# Generate Jupyter plugin settings
/accelerated-computing-hub/brev/jupyter-generate-plugin-settings.bash

mkdir -p /accelerated-computing-hub/logs

# Forward ports from other Docker Compose services to localhost so that
# jupyter-server-proxy can reach them (it only proxies to localhost).
if [ -n "${ACH_PORT_FORWARDS:-}" ]; then
  for FORWARD in ${ACH_PORT_FORWARDS:-}; do
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
