#!/bin/bash
set -e

# Install CUDA Toolkit
curl -sSL https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-debian12-12-9-local_12.9.0-575.51.03-1_amd64.deb -o /tmp/cuda-repo.deb
dpkg -i /tmp/cuda-repo.deb
cp /var/cuda-repo-debian12-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-9
rm -f /tmp/cuda-repo.deb

# Install Python
runuser -u ubuntu -- bash -lc '
set -e
cd ~
curl -sSL https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.13 --default --preview-features python-install-default
~/.local/bin/uv venv --seed ~/pyhpc-tutorial/.venv
printf "VIRTUAL_ENV_DISABLE_PROMPT=1 source ~/pyhpc-tutorial/.venv/bin/activate\n" >> ~/.bashrc
source ~/pyhpc-tutorial/.venv/bin/activate
pip install -r ~/pyhpc-tutorial/build/requirements.txt

mkdir -p ~/.jupyter/lab/user-settings/jupyterlab-nvidia-nsight
ln -fs ~/pyhpc-tutorial/build/jupyter_server_config.py ~/.jupyter/jupyter_server_config.py
ln -fs ~/pyhpc-tutorial/build/jupyter_nsight_plugin_settings.json ~/.jupyter/lab/user-settings/jupyterlab-nvidia-nsight/plugin.jupyterlab-settings
'

# Create the Jupyter service
cat >/etc/systemd/system/jupyterlab.service <<'EOF'
[Unit]
Description=JupyterLab
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/pyhpc-tutorial/notebooks
Environment=HOME=/home/ubuntu
ExecStart=/bin/bash -lc 'source /home/ubuntu/pyhpc-tutorial/.venv/bin/activate; exec python -m jupyter lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token="" --NotebookApp.password="" --NotebookApp.default_url=""'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now jupyterlab.service
