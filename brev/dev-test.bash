#! /bin/bash
#
# Test a Docker Compose file with local repository mounted.
#
# This script sets up the development environment and then calls
# test-docker-compose.bash to validate Docker Compose configurations.
#
# Usage:
#   ./brev/dev-test.bash [--mount|--no-mount] <tutorial-name|docker-compose-file> [test-args...]
#
# Examples:
#   ./brev/dev-test.bash accelerated-python
#   ./brev/dev-test.bash accelerated-python test/test_notebooks.py -k "03"
#   ./brev/dev-test.bash --mount accelerated-python

set -eu

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") [--mount|--no-mount] <tutorial-name|docker-compose-file> [test-args...]

Test a Docker Compose file with local repository mounted.

Options:
  --mount       Bind-mount local repo into the container (live local files)
  --no-mount    Use image content only (default)

Arguments:
  tutorial-name          Name of tutorial (e.g., accelerated-python)
  docker-compose-file    Path to docker-compose.yml file
  test-args              Extra arguments forwarded to the tutorial's test.bash

Examples:
  $(basename "$0") accelerated-python
  $(basename "$0") accelerated-python test/test_notebooks.py -k "03"
  $(basename "$0") --mount accelerated-python

Requirements:
  - Docker and Docker Compose must be installed
EOF
    exit 1
}

# Parse --mount/--no-mount flag (default: no mount for tests)
MOUNT=false
if [ $# -gt 0 ]; then
    case "$1" in
        --mount)    MOUNT=true;  shift ;;
        --no-mount) MOUNT=false; shift ;;
    esac
fi

# Check argument
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Tutorial name or Docker Compose file path is required${NC}"
    usage
fi

ARG=$1
shift
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

source ${SCRIPT_PATH}/dev-common.bash
setup_dev_env "${REPO_ROOT}"

echo "================================================================================"
echo "Starting tests (calling test-docker-compose.bash)"
echo "================================================================================"
echo ""

# Change to repo directory and call test-docker-compose.bash
cd ${REPO_ROOT}
MOUNT_FLAG="--no-mount"
if [ "${MOUNT}" = "true" ]; then
    MOUNT_FLAG="--mount"
fi
${SCRIPT_PATH}/test-docker-compose.bash ${MOUNT_FLAG} "${ARG}" "$@"
