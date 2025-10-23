#!/bin/bash

/accelerated-computing-hub/brev/jupyter-generate-settings.bash

mkdir -p /accelerated-computing-hub/logs

exec python -m jupyter lab --LabApp.default_url="${1}"
