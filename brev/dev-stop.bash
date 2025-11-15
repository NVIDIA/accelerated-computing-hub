#! /bin/bash

# This script stops Docker containers for a tutorial.
#
# Usage:
#   ./dev-stop.bash <tutorial-name>
#
# Example:
#   ./dev-stop.bash accelerated-python

set -eu

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)
MOUNT=/tmp/ach-mount

# Check argument
if [ $# -ne 1 ]; then
    echo "Error: Tutorial name is required"
    echo "Usage: $0 <tutorial-name>"
    echo "Example: $0 accelerated-python"
    exit 1
fi

ACH_TUTORIAL=$1
ACH_TUTORIAL_PATH="${REPO_ROOT}/tutorials/${ACH_TUTORIAL}"
DOCKER_COMPOSE="${ACH_TUTORIAL_PATH}/brev/docker-compose.yml"

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

echo "Stopping tutorial: ${ACH_TUTORIAL}"
cd ${MOUNT}
docker compose -f ${DOCKER_COMPOSE} down

cd / # We've got to be somewhere that isn't the mount to unmount it.
sudo umount ${MOUNT}
sudo rmdir ${MOUNT} 2>/dev/null || true

echo "Tutorial ${ACH_TUTORIAL} stopped successfully!"
