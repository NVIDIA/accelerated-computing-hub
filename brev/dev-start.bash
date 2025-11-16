#! /bin/bash

# This script starts Docker containers for a tutorial.
#
# Usage:
#   ./dev-start.bash <tutorial-name>
#
# Example:
#   ./dev-start.bash accelerated-python

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

sudo mkdir -p ${MOUNT}
sudo bindfs --force-user=$(id -u) --force-group=$(id -g) \
            --create-for-user=$(id -u) --create-for-group=$(id -g) \
            ${REPO_ROOT} ${MOUNT}

# Remove any existing volumes to ensure they point to the correct location.
docker volume rm "${ACH_TUTORIAL}_accelerated-computing-hub" &>/dev/null || true
docker volume rm accelerated-computing-hub &>/dev/null || true

# Create the accelerated-computing-hub volume as a bind mount to the local repository.
docker volume create --driver local \
  --opt type=none \
  --opt o=bind \
  --opt device=${MOUNT} \
  "${ACH_TUTORIAL}_accelerated-computing-hub"

echo "Starting tutorial: ${ACH_TUTORIAL}"
cd ${MOUNT}
docker compose -f ${DOCKER_COMPOSE} up -d

echo "Tutorial ${ACH_TUTORIAL} started successfully!"
