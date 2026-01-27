#! /bin/bash
#
# Runtime user environment setup.
#
# This script sets up the user's home directory and configuration files
# at container startup. It looks up HOME from /etc/passwd, falling back
# to /tmp if the user doesn't exist.
#
# This script is sourced via BASH_ENV and shell startup scripts.


# Get current user info
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Set HOME if not already set, if it's "/" (Docker default for unknown UIDs),
# or if it's /root but we're not root
if [ -z "${HOME:-}" ] || [ "${HOME}" = "/" ] || { [ "${HOME}" = "/root" ] && [ "${CURRENT_UID}" != "0" ]; }; then
    if [ "${CURRENT_UID}" = "0" ]; then
        export HOME="/root"
    else
        # Look up home directory from passwd, fall back to /tmp if user doesn't exist
        PASSWD_HOME=$(getent passwd "${CURRENT_UID}" 2>/dev/null | cut -d: -f6)
        if [ -n "${PASSWD_HOME}" ]; then
            export HOME="${PASSWD_HOME}"
        else
            export HOME="/tmp/home-${CURRENT_UID}"
        fi
        mkdir -p "${HOME}"
    fi
fi

# Ensure home directory exists and is writable
mkdir -p "${HOME}"

# Setup Jupyter configuration
export JUPYTER_CONFIG_DIR="${HOME}/.jupyter"
export JUPYTER_DATA_DIR="${HOME}/.local/share/jupyter"
export IPYTHONDIR="${HOME}/.ipython"

mkdir -p "${JUPYTER_CONFIG_DIR}"
mkdir -p "${JUPYTER_DATA_DIR}"
mkdir -p "${IPYTHONDIR}/profile_default/startup"

# Link Jupyter server config if not already present
if [ ! -e "${JUPYTER_CONFIG_DIR}/jupyter_server_config.py" ]; then
    ln -sf /accelerated-computing-hub/brev/jupyter-server-config.py "${JUPYTER_CONFIG_DIR}/jupyter_server_config.py"
fi

# Link IPython startup scripts if not already present
if [ ! -e "${IPYTHONDIR}/profile_default/startup/00-add-cwd-to-path.py" ]; then
    ln -sf /accelerated-computing-hub/brev/ipython-startup-add-cwd-to-path.py "${IPYTHONDIR}/profile_default/startup/00-add-cwd-to-path.py"
fi

# Setup bash history directory
mkdir -p "${HOME}/.local/state"

# Setup Git safe directory (needs to be done per-user)
git config --global --add safe.directory "/accelerated-computing-hub" 2>/dev/null || true

# Ensure logs directory exists and is writable
mkdir -p /accelerated-computing-hub/logs 2>/dev/null || true
