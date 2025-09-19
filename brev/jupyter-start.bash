#!/bin/bash

/accelerated-computing-hub/brev/jupyter-generate-settings.bash

exec python -m jupyter lab \
  --allow-root \
  --ip=0.0.0.0 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --NotebookApp.default_url=''
