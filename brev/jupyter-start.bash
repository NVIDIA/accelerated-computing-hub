#!/bin/bash

/pyhpc-tutorial/build/jupyter-generate-nsight-plugin-settings.bash > ~/.jupyter/lab/user-settings/jupyterlab-nvidia-nsight/plugin.jupyterlab-settings

exec python -m jupyter lab \
  --allow-root \
  --ip=0.0.0.0 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --NotebookApp.default_url=''
