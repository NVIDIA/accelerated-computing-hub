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
