#!/bin/bash

/accelerated-computing-hub/brev/jupyter-generate-settings.bash

mkdir -p /accelerated-computing-hub/logs

ARGS=""
if [ -n "${1}" ]; then
  ARGS="--LabApp.default_url=${1}"
fi

exec python -m jupyter lab ${ARGS}
