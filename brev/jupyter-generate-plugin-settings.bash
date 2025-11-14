#! /bin/bash

set -eu

JUPYTER_HOST="jupyter0-${BREV_ENV_ID}.brevlab.com"
NSIGHT_HTTP_URL="https://nsight0-${BREV_ENV_ID}.brevlab.com"

# Theme
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
cat << EOF > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings
{
  "theme": "JupyterLab Dark",
  "adaptive-theme": true,
  "preferred-light-theme": "JupyterLab Light",
  "preferred-dark-theme": "JupyterLab Dark",
  "theme-scrollbars": false
}
EOF

# Nsight JupyterLab Plugin
mkdir -p ~/.jupyter/lab/user-settings/jupyterlab-nvidia-nsight
cat << EOF > ~/.jupyter/lab/user-settings/jupyterlab-nvidia-nsight/plugin.jupyterlab-settings
{
  "ui": {
    "enabled": true,
    "suppressServerAddressWarning": true,
    "host": "${JUPYTER_HOST}",
    "dockerHost": "${JUPYTER_HOST}",
    "defaultStreamerAddress": "${NSIGHT_HTTP_URL}"
  }
}
EOF
