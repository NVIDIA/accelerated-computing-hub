#! /bin/bash

set -eu

apt-get update
apt-get install -y curl

EXTERNAL_IP=$(curl -sSL ifconfig.me)

export NVIDIA_DRIVER_CAPABILITIES=all
export HOST_IP=0.0.0.0
export WEB_USERNAME=""
export WEB_PASSWORD=""
export SELKIES_TURN_HOST=$(curl -sSL ifconfig.me)

VARS=("NVIDIA_DRIVER_CAPABILITIES" "HOST_IP" "WEB_USERNAME" "WEB_PASSWORD" "SELKIES_TURN_HOST")
for VAR in "${VARS[@]}"; do
  if [[ -n "${!VAR+x}" ]]; then
    echo "export $VAR=${!VAR}" >> ~/.bashrc
  fi
done

# Workaround: The Nsight Streamer container isn't restartable because it unconditionally creates
# symlinks every time its start, which fails if the symlinks already exists.
if test -h /usr/lib/x86_64-linux-gnu/libnvrtc.so; then
  rm /usr/lib/x86_64-linux-gnu/libnvrtc.so
fi

if test -d /mnt/persist/home/host; then
  rm /mnt/persist/home/host
fi

source /setup/entrypoint.sh "$@"
