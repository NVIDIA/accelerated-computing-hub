#! /bin/bash
#
# Entrypoint for the nsight service. Runs as root to do setup, then hands off
# to the nsight streamer's own entrypoint which handles user switching.

set -euo pipefail

# Set USER for nsight streamer's internal user switching
export USER="${ACH_TARGET_USER}"
export HOME="${ACH_TARGET_HOME}"

# Ensure a group with the user's name exists.
# The Nsight Streamer's entrypoint does `chown $USER:$USER` which requires a group
# with the same name as the user. Our main entrypoint may have reused an existing
# group with a different name if the target GID was already taken.
if ! getent group "${ACH_TARGET_USER}" &>/dev/null; then
    groupadd "${ACH_TARGET_USER}" || true
fi

# Copy Nsight config and bashrc from nvidia user to target user's home
mkdir -p "${ACH_TARGET_HOME}/.config/NVIDIA Corporation"
cp "/home/nvidia/.config/NVIDIA Corporation/NVIDIA Nsight Systems.ini" "${ACH_TARGET_HOME}/.config/NVIDIA Corporation/"
cp /home/nvidia/.bashrc "${ACH_TARGET_HOME}/.bashrc"
chown -R "${ACH_TARGET_USER}:$(id -gn ${ACH_TARGET_USER})" "${ACH_TARGET_HOME}"

# Install curl if not present
if ! command -v curl &> /dev/null; then
    apt-get update
    apt-get install -y curl
fi

# Set environment variables for Nsight Streamer
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
# symlinks every time it starts, which fails if the symlinks already exist.
if test -e /usr/lib/x86_64-linux-gnu/libnvrtc.so; then
  rm -f /usr/lib/x86_64-linux-gnu/libnvrtc.so
fi

if test -e /mnt/persist/home/host; then
  rm -rf /mnt/persist/home/host
fi

# Hand off to nsight streamer's entrypoint (which handles user switching via USER env var)
source /setup/entrypoint.sh "$@"
