#! /bin/bash

# Generate Jupyter plugin settings
/accelerated-computing-hub/brev/jupyter-generate-plugin-settings.bash

mkdir -p /accelerated-computing-hub/logs

# Set the preferred directory to the current working directory, which is set by Docker Compose.
ARGS="--ServerApp.preferred_dir=${PWD:-/}"

if [ -n "${1}" ]; then
  ARGS="--LabApp.default_url=${1}"
fi

exec python -m jupyter lab ${ARGS}
