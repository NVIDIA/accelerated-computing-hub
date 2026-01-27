#! /bin/bash
#
# Helper functions for setting up development mounts with Docker's --user flag.
#
# This library provides functions for:
# - Exporting HOST_UID/HOST_GID for Docker to match host user permissions
# - Creating Docker volumes pointing to the repository
#
# Usage:
#   source ./brev/dev-common.bash
#   setup_dev_env "/path/to/repo"
#   create_docker_volume "tutorial-name"

# Export host user's UID/GID/username for Docker Compose
setup_dev_env() {
    local REPO_ROOT=${1}

    echo "🔧 Setting up development environment..."
    export ACH_UID=$(id -u)
    export ACH_GID=$(id -g)
    export ACH_USER=$(id -un)
    export ACH_REPO_ROOT="${REPO_ROOT}"
    echo -e "${GREEN:-}✅ ACH_UID=${ACH_UID}, ACH_GID=${ACH_GID}, ACH_USER=${ACH_USER}${NC:-}"
    echo ""
}

# Create user and switch to them if running as root.
# Arguments:
#   $1 - "login" for interactive login shell, "exec" to re-exec calling script
# If not root, does nothing.
create_user_and_switch() {
    local MODE="${1:-exec}"

    if [ "$(id -u)" != "0" ]; then
        return
    fi

    local TARGET_USER="${ACH_USER:-ach}"
    local TARGET_GID="${ACH_GID:-1000}"

    if ! id "${TARGET_USER}" &>/dev/null; then
        # Check if a group with the target GID already exists
        local EXISTING_GROUP=$(getent group "${TARGET_GID}" 2>/dev/null | cut -d: -f1)
        if [ -n "${EXISTING_GROUP}" ]; then
            local TARGET_GROUP="${EXISTING_GROUP}"
        else
            groupadd --gid "${TARGET_GID}" "${TARGET_USER}"
            local TARGET_GROUP="${TARGET_USER}"
        fi
        useradd --uid "${ACH_UID:-1000}" --gid "${TARGET_GROUP}" --create-home --shell /bin/bash "${TARGET_USER}"
        getent group docker &>/dev/null && usermod -aG docker "${TARGET_USER}"
    fi

    local TARGET_HOME=$(getent passwd "${TARGET_USER}" | cut -d: -f6)
    export HOME="${TARGET_HOME}"

    if [ "${MODE}" = "login" ]; then
        exec su -l "${TARGET_USER}"
    else
        exec su -s /bin/bash -p "${TARGET_USER}" -c "$0 $*"
    fi
}

# Create Docker volume pointing to the repository
create_docker_volume() {
    local ACH_TUTORIAL=${1}

    # Remove any existing volumes to ensure they point to the correct location
    echo "🗑️  Removing any existing Docker volumes..."
    docker volume rm "${ACH_TUTORIAL}_accelerated-computing-hub" &>/dev/null || true
    docker volume rm accelerated-computing-hub &>/dev/null || true
    echo ""

    # Create the accelerated-computing-hub volume as a bind mount to the local repository
    echo "🔧 Creating Docker volume for ${ACH_TUTORIAL}..."
    docker volume create --driver local \
      --opt type=none \
      --opt o=bind \
      --opt device="${ACH_REPO_ROOT}" \
      "${ACH_TUTORIAL}_accelerated-computing-hub"
    echo -e "${GREEN:-}✅ Docker volume created${NC:-}"
    echo ""
}
