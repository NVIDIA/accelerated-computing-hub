#! /bin/bash

set -eu

# Install curl if not present (needs sudo for non-root users)
if ! command -v curl &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y curl
fi

EXTERNAL_IP=$(curl -sSL ifconfig.me)

export NVIDIA_DRIVER_CAPABILITIES=all
export HOST_IP=0.0.0.0
export WEB_USERNAME=""
export WEB_PASSWORD=""
export SELKIES_TURN_HOST=$(curl -sSL ifconfig.me)

VARS=("NVIDIA_DRIVER_CAPABILITIES" "HOST_IP" "WEB_USERNAME" "WEB_PASSWORD" "SELKIES_TURN_HOST")
for VAR in "${VARS[@]}"; do
  if [[ -n "${!VAR+x}" ]]; then
    echo "export $VAR=${!VAR}" >> "${HOME}/.bashrc"
  fi
done

# Workaround: The Nsight Streamer container isn't restartable because it unconditionally creates
# symlinks every time its start, which fails if the symlinks already exists.
if test -h /usr/lib/x86_64-linux-gnu/libnvrtc.so; then
  sudo rm /usr/lib/x86_64-linux-gnu/libnvrtc.so
fi

if test -d /mnt/persist/home/host; then
  sudo rm /mnt/persist/home/host
fi

source /setup/entrypoint.sh "$@"
