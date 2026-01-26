#! /bin/bash

set -eu

# Ensure HOME is set (user-setup.bash should have done this, but be safe)
if [ -z "${HOME:-}" ]; then
    export HOME="/home/ach"
fi

JUPYTER_HOST="jupyter0-${BREV_ENV_ID:-local}.brevlab.com"
NSIGHT_HTTP_URL="https://nsight0-${BREV_ENV_ID:-local}.brevlab.com"

# Theme
mkdir -p "${HOME}/.jupyter/lab/user-settings/@jupyterlab/apputils-extension"
cat << EOF > "${HOME}/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings"
{
  "theme": "JupyterLab Dark",
  "adaptive-theme": true,
  "preferred-light-theme": "JupyterLab Light",
  "preferred-dark-theme": "JupyterLab Dark",
  "theme-scrollbars": false
}
EOF

# Nsight JupyterLab Plugin
mkdir -p "${HOME}/.jupyter/lab/user-settings/jupyterlab-nvidia-nsight"
cat << EOF > "${HOME}/.jupyter/lab/user-settings/jupyterlab-nvidia-nsight/plugin.jupyterlab-settings"
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

# Execution timing
mkdir -p "${HOME}/.jupyter/lab/user-settings/@jupyterlab/notebook-extension"
cat << EOF > "${HOME}/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings"
{
  "recordTiming": true
}
EOF
