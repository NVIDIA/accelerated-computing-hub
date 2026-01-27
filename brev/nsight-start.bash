#! /bin/bash

set -eu

# Set USER from ACH_USER for nsight streamer
export USER="${ACH_USER:-ach}"

# Create user if they don't exist (for matching Jupyter container user)
if ! id "${USER}" &>/dev/null; then
    TARGET_GID="${ACH_GID:-1000}"
    # Check if a group with the target GID already exists
    EXISTING_GROUP=$(getent group "${TARGET_GID}" 2>/dev/null | cut -d: -f1)
    if [ -n "${EXISTING_GROUP}" ]; then
        TARGET_GROUP="${EXISTING_GROUP}"
    else
        groupadd --gid "${TARGET_GID}" "${USER}"
        TARGET_GROUP="${USER}"
    fi
    useradd --uid "${ACH_UID:-1000}" --gid "${TARGET_GROUP}" --create-home --shell /bin/bash "${USER}"

    # Copy Nsight config and bashrc from nvidia user
    USER_HOME="/home/${USER}"
    mkdir -p "${USER_HOME}/.config/NVIDIA Corporation"
    cp "/home/nvidia/.config/NVIDIA Corporation/NVIDIA Nsight Systems.ini" "${USER_HOME}/.config/NVIDIA Corporation/"
    cp /home/nvidia/.bashrc "${USER_HOME}/.bashrc"
    chown -R "${USER}:${TARGET_GROUP}" "${USER_HOME}"
fi

# Install curl if not present
if ! command -v curl &> /dev/null; then
    apt-get update
    apt-get install -y curl
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
  rm /usr/lib/x86_64-linux-gnu/libnvrtc.so
fi

if test -d /mnt/persist/home/host; then
  rm /mnt/persist/home/host
fi

source /setup/entrypoint.sh "$@"
