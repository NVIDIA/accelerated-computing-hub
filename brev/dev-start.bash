#! /bin/bash

# This script starts Docker containers for a tutorial.
#
# Usage:
#   ./dev-start.bash [--mount|--no-mount] <tutorial-name>
#
# Examples:
#   ./dev-start.bash accelerated-python              # bind-mounts local repo (default)
#   ./dev-start.bash --no-mount accelerated-python   # uses image content only

set -eu

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Parse --mount/--no-mount flag (default: mount for dev-start)
MOUNT=true
if [ $# -gt 0 ]; then
    case "$1" in
        --mount)    MOUNT=true;  shift ;;
        --no-mount) MOUNT=false; shift ;;
    esac
fi

# Check argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 [--mount|--no-mount] <tutorial-name>"
    echo ""
    echo "Options:"
    echo "  --mount       Bind-mount local repo into the container (default)"
    echo "  --no-mount    Use image content only"
    echo ""
    echo "Example: $0 accelerated-python"
    exit 1
fi

ACH_TUTORIAL=$1
ACH_TUTORIAL_PATH="${REPO_ROOT}/tutorials/${ACH_TUTORIAL}"
DOCKER_COMPOSE="${ACH_TUTORIAL_PATH}/brev/docker-compose.yml"
DOCKER_COMPOSE_DEV="/tmp/docker-compose.${ACH_TUTORIAL}.dev.yml"

# Validate tutorial exists
if [ ! -d "${ACH_TUTORIAL_PATH}" ]; then
    echo "Error: Tutorial directory not found: ${ACH_TUTORIAL_PATH}"
    exit 1
fi

# Validate docker-compose.yml exists
if [ ! -f "${DOCKER_COMPOSE}" ]; then
    echo "Error: No docker-compose.yml found at ${DOCKER_COMPOSE}"
    exit 1
fi

source ${SCRIPT_PATH}/dev-common.bash
setup_dev_env "${REPO_ROOT}"
setup_docker_volume "${ACH_TUTORIAL}" "${MOUNT}"

echo "Starting tutorial: ${ACH_TUTORIAL}"
cd ${REPO_ROOT}

# Create a modified docker-compose file that binds to 0.0.0.0 instead of 127.0.0.1
# This is needed for local development so services are accessible from outside the container
sed 's/127\.0\.0\.1:/0.0.0.0:/g' "${DOCKER_COMPOSE}" > "${DOCKER_COMPOSE_DEV}"

# Filter out the "volume already exists" warning while preserving all other warnings/errors on stderr
docker compose -f ${DOCKER_COMPOSE_DEV} up -d 2> >(grep -v "already exists but was not created by Docker Compose" >&2)

echo "Tutorial ${ACH_TUTORIAL} started successfully!"
