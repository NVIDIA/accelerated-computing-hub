#! /bin/bash

# This script starts Docker containers for a tutorial.
#
# Usage:
#   ./dev-start.bash <tutorial-name>
#
# Example:
#   ./dev-start.bash accelerated-python

set -euo pipefail

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

TUTORIAL_NAME=$1
TUTORIAL_PATH="${REPO_ROOT}/tutorials/${TUTORIAL_NAME}"
DOCKER_COMPOSE="${TUTORIAL_PATH}/brev/docker-compose.yml"

# Validate tutorial exists
if [ ! -d "${TUTORIAL_PATH}" ]; then
    echo "Error: Tutorial directory not found: ${TUTORIAL_PATH}"
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

# Create the accelerated-computing-hub volume as a bind mount to the repository
# Remove existing volume if it exists to ensure it points to the correct location
docker volume rm accelerated-computing-hub 2>/dev/null || true
docker volume create --driver local \
  --opt type=none \
  --opt o=bind \
  --opt device=${MOUNT} \
  accelerated-computing-hub

echo "Starting tutorial: ${TUTORIAL_NAME}"
cd ${MOUNT}
docker compose -f ${DOCKER_COMPOSE} up -d

echo "Tutorial ${TUTORIAL_NAME} started successfully!"
