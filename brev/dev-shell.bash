#! /bin/bash

# This script starts an interactive shell in a Docker container for a tutorial.
#
# Usage:
#   ./brev/dev-shell.bash <tutorial-name|docker-compose-file> <service-name>
#
# Examples:
#   ./brev/dev-shell.bash accelerated-python base
#   ./brev/dev-shell.bash accelerated-python jupyter
#   ./brev/dev-shell.bash tutorials/accelerated-python/brev/docker-compose.yml jupyter

set -eu

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)

source "${SCRIPT_PATH}/dev-common.bash"
setup_dev_env "${REPO_ROOT}"

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") <tutorial-name|docker-compose-file> <service-name>

Start an interactive shell in a Docker container for a tutorial.

Arguments:
  tutorial-name          Name of tutorial (e.g., accelerated-python)
  docker-compose-file    Path to docker-compose.yml file
  service-name           Name of service to run (e.g., base, jupyter)

Examples:
  $(basename "$0") accelerated-python base
  $(basename "$0") accelerated-python jupyter
  $(basename "$0") tutorials/accelerated-python/brev/docker-compose.yml jupyter

Requirements:
  - Docker and Docker Compose must be installed
EOF
    exit 1
}

# Check arguments
if [ $# -ne 2 ]; then
    echo -e "${RED}Error: Tutorial name/file and service name are required${NC}"
    usage
fi

ARG=$1
SERVICE=$2
COMPOSE_FILE=""
ACH_TUTORIAL=""

# Check if the argument is a file path (contains / or is an absolute path)
if [[ "${ARG}" == *"/"* ]]; then
    # Treat as a file path
    COMPOSE_FILE="${ARG}"

    # Convert to absolute path if relative
    if [[ "${COMPOSE_FILE}" != /* ]]; then
        COMPOSE_FILE="${REPO_ROOT}/${COMPOSE_FILE}"
    fi

    # Extract tutorial name from path: .../tutorials/<tutorial-name>/brev/docker-compose.yml
    COMPOSE_DIR=$(dirname "${COMPOSE_FILE}")
    TUTORIAL_DIR=$(dirname "${COMPOSE_DIR}")
    ACH_TUTORIAL=$(basename "${TUTORIAL_DIR}")
else
    # Treat as a tutorial name
    ACH_TUTORIAL="${ARG}"
    ACH_TUTORIAL_PATH="${REPO_ROOT}/tutorials/${ACH_TUTORIAL}"

    # Check if tutorial directory exists
    if [ ! -d "${ACH_TUTORIAL_PATH}" ]; then
        echo -e "${RED}Error: Tutorial directory not found: ${ACH_TUTORIAL_PATH}${NC}"
        exit 1
    fi

    COMPOSE_FILE="${ACH_TUTORIAL_PATH}/brev/docker-compose.yml"
fi

# Validate docker-compose file exists
if [ ! -f "${COMPOSE_FILE}" ]; then
    echo -e "${RED}Error: Docker Compose file not found: ${COMPOSE_FILE}${NC}"
    exit 1
fi

echo "================================================================================"
echo "Starting interactive shell for: ${ACH_TUTORIAL} (service: ${SERVICE})"
echo "Docker Compose file: ${COMPOSE_FILE}"
echo "================================================================================"
echo ""

# Check for and remove existing volume to ensure fresh state
VOLUME_NAME="${ACH_TUTORIAL}_accelerated-computing-hub"
if docker volume inspect "${VOLUME_NAME}" &>/dev/null; then
    echo "🗑️  Removing existing volume: ${VOLUME_NAME}"
    docker volume rm "${VOLUME_NAME}" &>/dev/null || true
    echo ""
fi

# Run interactive shell with user switching
echo "🚀 Starting interactive shell as ${ACH_USER}..."
echo "   (Type 'exit' or press Ctrl+D to exit the shell)"
echo ""

docker compose -f "${COMPOSE_FILE}" run --rm -it \
    --entrypoint "/accelerated-computing-hub/brev/entrypoint.bash shell"

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Shell session ended${NC}"
echo "================================================================================"
