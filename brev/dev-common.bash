#! /bin/bash
#
# Helper functions for setting up development mounts with Docker's --user flag.
#
# This library provides functions for:
# - Exporting HOST_UID/HOST_GID for Docker to match host user permissions
# - Creating Docker volumes (bind-mount to local repo or plain image-backed)
#
# Usage:
#   source ./brev/dev-common.bash
#   setup_dev_env "/path/to/repo"
#   setup_docker_volume "tutorial-name" true   # bind-mount local repo
#   setup_docker_volume "tutorial-name" false  # image content only

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

# Set up the Docker volume for a tutorial.
#
# Always removes any existing volume first for a clean state.
# When MOUNT=true, creates a bind-mount volume pointing to the local repo.
# When MOUNT=false, does nothing (docker compose will create a plain volume
# populated from the image on first use).
setup_docker_volume() {
    local ACH_TUTORIAL=${1}
    local MOUNT=${2:-true}
    local VOLUME_NAME="${ACH_TUTORIAL}_accelerated-computing-hub"

    # Remove existing volume for clean state
    if docker volume inspect "${VOLUME_NAME}" &>/dev/null; then
        echo "🗑️  Removing existing Docker volume: ${VOLUME_NAME}"
        docker volume rm "${VOLUME_NAME}" &>/dev/null || true
    fi

    if [ "${MOUNT}" = "true" ]; then
        echo "🔧 Creating Docker volume (bind mount to local repo)..."
        docker volume create --driver local \
          --opt type=none \
          --opt o=bind \
          --opt device="${ACH_REPO_ROOT}" \
          "${ACH_TUTORIAL}_accelerated-computing-hub" > /dev/null
        echo -e "${GREEN:-}✅ Docker volume created (local mount)${NC:-}"
    else
        echo "📦 Using image content (no local mount)"
    fi
    echo ""
}

# Kept for backward compatibility — equivalent to setup_docker_volume with mount=true.
create_docker_volume() {
    setup_docker_volume "${1}" true
}
