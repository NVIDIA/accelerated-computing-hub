#! /bin/bash
#
# Helper functions for setting up development mounts with bindfs.
#
# This library provides functions for:
# - Setting up bindfs mounts of the repository
# - Creating Docker volumes pointing to the mount
# - Cleaning up mounts
#
# Usage:
#   source ./brev/dev-mount.bash
#   setup_dev_mount
#   create_docker_volume "tutorial-name"
#   # ... do work ...
#   cleanup_dev_mount

# Mount location
MOUNT=/tmp/accelerated-computing-hub-mount

# Setup bindfs mount for the repository
setup_dev_mount() {
    local REPO_ROOT=${1}

    echo "🔧 Setting up bindfs mount at ${MOUNT}..."
    sudo mkdir -p ${MOUNT}
    sudo bindfs --force-user=$(id -u) --force-group=$(id -g) \
                --create-for-user=$(id -u) --create-for-group=$(id -g) \
                ${REPO_ROOT} ${MOUNT}
    echo -e "${GREEN:-}✅ Bindfs mount created${NC:-}"
    echo ""
}

# Create Docker volume pointing to the mount
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
      --opt device=${MOUNT} \
      "${ACH_TUTORIAL}_accelerated-computing-hub"
    echo -e "${GREEN:-}✅ Docker volume created${NC:-}"
    echo ""
}

# Cleanup bindfs mount
cleanup_dev_mount() {
    echo ""
    echo "🧹 Cleaning up mount..."

    # Change to a directory outside the mount before unmounting
    cd /

    # Unmount bindfs
    if mountpoint -q ${MOUNT} 2>/dev/null; then
        echo "🔧 Unmounting bindfs from ${MOUNT}..."
        sudo umount ${MOUNT} || sudo umount -l ${MOUNT}
        echo -e "${GREEN:-}✅ Bindfs unmounted${NC:-}"
    fi
    echo ""
}

# Setup cleanup trap (call this to register automatic cleanup on exit)
setup_cleanup_trap() {
    trap 'EXIT_CODE=$?; cleanup_dev_mount; exit ${EXIT_CODE}' EXIT INT TERM
}
