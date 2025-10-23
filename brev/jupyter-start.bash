#!/bin/bash

/accelerated-computing-hub/brev/jupyter-generate-settings.bash

mkdir -p /accelerated-computing-hub/logs

exec python -m jupyter lab --ServerApp.default_url="${1}"
