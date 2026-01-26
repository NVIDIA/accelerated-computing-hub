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

source ${SCRIPT_PATH}/dev-mount.bash

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

setup_dev_env "${REPO_ROOT}"
create_docker_volume "${ACH_TUTORIAL}"

echo "Starting tutorial: ${ACH_TUTORIAL}"
cd ${REPO_ROOT}

# Create a modified docker-compose file that binds to 0.0.0.0 instead of 127.0.0.1
# This is needed for local development so services are accessible from outside the container
sed 's/127\.0\.0\.1:/0.0.0.0:/g' "${DOCKER_COMPOSE}" > "${DOCKER_COMPOSE_DEV}"

# Filter out the "volume already exists" warning while preserving all other warnings/errors on stderr
docker compose -f ${DOCKER_COMPOSE_DEV} up -d 2> >(grep -v "already exists but was not created by Docker Compose" >&2)

echo "Tutorial ${ACH_TUTORIAL} started successfully!"
