#! /bin/bash
#
# Main entrypoint for all services. Creates the user if needed, then dispatches
# to the service-specific entrypoint script.
#
# Usage: entrypoint.bash <service> [args...]
#   service: base, jupyter, nsight, shell

set -euo pipefail

SERVICE="${1:-}"
shift || true

if [ -z "${SERVICE}" ]; then
    echo "Error: No service specified. Usage: entrypoint.bash <service> [args...]" >&2
    exit 1
fi

# Install gosu if not present
if ! command -v gosu &> /dev/null; then
    apt-get update -y
    apt-get install -y gosu
fi

# Create user if running as root and user doesn't exist
if [ "$(id -u)" = "0" ]; then
    TARGET_USER="${ACH_USER:-ach}"
    TARGET_UID="${ACH_UID:-1000}"
    TARGET_GID="${ACH_GID:-1000}"

    if ! id "${TARGET_USER}" &>/dev/null; then
        # Check if a user with the target UID already exists.
        EXISTING_USER=$(getent passwd "${TARGET_UID}" 2>/dev/null | cut -d: -f1 || true)
        if [ -n "${EXISTING_USER}" ]; then
            TARGET_USER="${EXISTING_USER}"
        else
            # Check if a group with the target GID already exists
            EXISTING_GROUP=$(getent group "${TARGET_GID}" 2>/dev/null | cut -d: -f1 || true)
            if [ -n "${EXISTING_GROUP}" ]; then
                TARGET_GROUP="${EXISTING_GROUP}"
            else
                groupadd --gid "${TARGET_GID}" "${TARGET_USER}"
                TARGET_GROUP="${TARGET_USER}"
            fi
            useradd --uid "${TARGET_UID}" --gid "${TARGET_GROUP}" --create-home --shell /bin/bash "${TARGET_USER}"
            getent group docker &>/dev/null && usermod -aG docker "${TARGET_USER}"
        fi
    fi

    # Export for use by service entrypoints
    export ACH_TARGET_USER="${TARGET_USER}"
    export ACH_TARGET_HOME=$(getent passwd "${TARGET_USER}" | cut -d: -f6 || true)
    # Setup user environment (one-time setup, not on every shell)
    export HOME="${ACH_TARGET_HOME}"

    # Setup Jupyter configuration directories
    gosu "${TARGET_USER}" mkdir -p "${HOME}/.jupyter"
    gosu "${TARGET_USER}" mkdir -p "${HOME}/.local/share/jupyter"
    gosu "${TARGET_USER}" mkdir -p "${HOME}/.ipython/profile_default/startup"
    gosu "${TARGET_USER}" mkdir -p "${HOME}/.local/state"

    # Link Jupyter server config if not already present
    if [ ! -e "${HOME}/.jupyter/jupyter_server_config.py" ]; then
        gosu "${TARGET_USER}" ln -sf /accelerated-computing-hub/brev/jupyter-server-config.py "${HOME}/.jupyter/jupyter_server_config.py"
    fi

    # Link IPython startup scripts if not already present
    if [ ! -e "${HOME}/.ipython/profile_default/startup/00-add-cwd-to-path.py" ]; then
        gosu "${TARGET_USER}" ln -sf /accelerated-computing-hub/brev/ipython-startup-add-cwd-to-path.py "${HOME}/.ipython/profile_default/startup/00-add-cwd-to-path.py"
    fi

    # Setup Git safe directory (run as target user)
    gosu "${TARGET_USER}" git config --global --add safe.directory "/accelerated-computing-hub" 2>/dev/null || true

    # Ensure logs directory exists
    gosu "${TARGET_USER}" mkdir -p /accelerated-computing-hub/logs
fi

# Dispatch to service-specific entrypoint
SERVICE_ENTRYPOINT="/accelerated-computing-hub/brev/entrypoint-${SERVICE}.bash"
if [ ! -x "${SERVICE_ENTRYPOINT}" ]; then
    echo "Error: Service entrypoint not found or not executable: ${SERVICE_ENTRYPOINT}" >&2
    exit 1
fi

exec "${SERVICE_ENTRYPOINT}" "$@"
