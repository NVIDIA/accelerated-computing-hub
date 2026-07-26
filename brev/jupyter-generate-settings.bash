#! /bin/bash

set -eu

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

# Execution timing
mkdir -p "${HOME}/.jupyter/lab/user-settings/@jupyterlab/notebook-extension"
cat << EOF > "${HOME}/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings"
{
  "recordTiming": true
}
EOF
