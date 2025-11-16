#! /bin/bash
#
# Test a Docker Compose file with local repository mounted.
#
# This script sets up bindfs mount (like dev-start.bash) and then calls
# test-docker-compose.bash to validate Docker Compose configurations.
#
# Usage:
#   ./brev/dev-test.bash <tutorial-name|docker-compose-file>
#
# Examples:
#   ./brev/dev-test.bash accelerated-python
#   ./brev/dev-test.bash tutorials/accelerated-python/brev/docker-compose.yml

set -eu

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)

source ${SCRIPT_PATH}/dev-mount.bash

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") <tutorial-name|docker-compose-file>

Test a Docker Compose file with local repository mounted.

Arguments:
  tutorial-name          Name of tutorial (e.g., accelerated-python)
  docker-compose-file    Path to docker-compose.yml file

Examples:
  $(basename "$0") accelerated-python
  $(basename "$0") tutorials/accelerated-python/brev/docker-compose.yml

Requirements:
  - Docker and Docker Compose must be installed
  - bindfs must be installed (sudo apt-get install bindfs)
EOF
    exit 1
}

# Check argument
if [ $# -ne 1 ]; then
    echo -e "${RED}Error: Tutorial name or Docker Compose file path is required${NC}"
    usage
fi

ARG=$1
ACH_TUTORIAL=""

# Determine tutorial name
if [[ "${ARG}" == *"/"* ]]; then
    # Treat as a file path - extract tutorial name
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
fi

echo "================================================================================"
echo "Setting up development environment for testing"
echo "================================================================================"
echo ""

# Setup mount and cleanup trap
setup_cleanup_trap
setup_dev_mount "${REPO_ROOT}"
create_docker_volume "${ACH_TUTORIAL}"

echo "================================================================================"
echo "Starting tests (calling test-docker-compose.bash)"
echo "================================================================================"
echo ""

# Change to mount directory and call test-docker-compose.bash
cd ${MOUNT}
${SCRIPT_PATH}/test-docker-compose.bash "${ARG}"
