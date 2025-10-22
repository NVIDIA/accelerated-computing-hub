#!/bin/bash

/accelerated-computing-hub/brev/jupyter-generate-settings.bash

exec python -m jupyter lab --NotebookApp.default_url="${1}"
