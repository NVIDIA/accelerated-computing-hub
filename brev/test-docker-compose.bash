#!/usr/bin/env bash
#
# Test a Docker Compose file by starting and stopping containers.
#
# This script validates Docker Compose configurations by attempting to start,
# inspect, and cleanly stop containers.
#
# Usage:
#   ./brev/test-docker-compose.bash <tutorial-name|docker-compose-file>
#
# Examples:
#   ./brev/test-docker-compose.bash accelerated-python
#   ./brev/test-docker-compose.bash tutorials/accelerated-python/brev/docker-compose.yml

set -euo pipefail

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
Usage: $(basename "$0") <tutorial-name|docker-compose-file>

Test a Docker Compose file by starting and stopping containers.

Arguments:
  tutorial-name          Name of tutorial (e.g., accelerated-python)
  docker-compose-file    Path to docker-compose.yml file

Examples:
  $(basename "$0") accelerated-python
  $(basename "$0") tutorials/accelerated-python/brev/docker-compose.yml

Requirements:
  - Docker and Docker Compose must be installed
EOF
    exit 1
}

# Check argument
if [ $# -ne 1 ]; then
    echo -e "${RED}Error: Tutorial name or Docker Compose file path is required${NC}"
    usage
fi

ARG=$1
COMPOSE_FILE=""

# Check if the argument is a file path (contains / or is an absolute path)
if [[ "${ARG}" == *"/"* ]]; then
    # Treat as a file path
    COMPOSE_FILE="${ARG}"

    # Convert to absolute path if relative
    if [[ "${COMPOSE_FILE}" != /* ]]; then
        COMPOSE_FILE="${REPO_ROOT}/${COMPOSE_FILE}"
    fi
else
    # Treat as a tutorial name
    TUTORIAL_NAME="${ARG}"
    TUTORIAL_PATH="${REPO_ROOT}/tutorials/${TUTORIAL_NAME}"

    # Check if tutorial directory exists
    if [ ! -d "${TUTORIAL_PATH}" ]; then
        echo -e "${RED}Error: Tutorial directory not found: ${TUTORIAL_PATH}${NC}"
        exit 1
    fi

    COMPOSE_FILE="${TUTORIAL_PATH}/brev/docker-compose.yml"
fi

# Validate docker-compose file exists
if [ ! -f "${COMPOSE_FILE}" ]; then
    echo -e "${RED}Error: Docker Compose file not found: ${COMPOSE_FILE}${NC}"
    exit 1
fi

echo "================================================================================"
echo "Testing Docker Compose: ${COMPOSE_FILE}"
echo "================================================================================"
echo ""

# Start containers
echo "📦 Starting containers..."
echo ""
if docker compose -f "${COMPOSE_FILE}" up -d; then
    echo ""
    echo -e "${GREEN}✅ Containers started successfully${NC}"
    echo ""

    # Wait a moment for containers to initialize
    echo "⏳ Waiting for containers to initialize..."
    sleep 5
    echo ""

    # Show container status
    echo "📊 Container status:"
    docker compose -f "${COMPOSE_FILE}" ps
    echo ""

    # Capture and display logs
    echo "📋 Container logs:"
    echo "--------------------------------------------------------------------------------"
    docker compose -f "${COMPOSE_FILE}" logs
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Stop containers
    echo "🛑 Stopping containers..."
    if docker compose -f "${COMPOSE_FILE}" down; then
        echo -e "${GREEN}✅ Containers stopped successfully${NC}"
        echo ""
        return_code=0
    else
        echo -e "${RED}❌ Failed to stop containers${NC}"
        echo ""
        return_code=1
    fi
else
    echo ""
    echo -e "${RED}❌ Failed to start containers${NC}"
    echo ""

    # Try to capture any logs that might be available
    echo "📋 Attempting to capture logs from failed startup:"
    echo "--------------------------------------------------------------------------------"
    docker compose -f "${COMPOSE_FILE}" logs || true
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Try to clean up
    echo "🛑 Attempting cleanup..."
    docker compose -f "${COMPOSE_FILE}" down || true
    echo ""

    return_code=1
fi

echo ""
echo "================================================================================"
if [ ${return_code} -eq 0 ]; then
    echo -e "${GREEN}✅ TEST PASSED: ${COMPOSE_FILE}${NC}"
else
    echo -e "${RED}❌ TEST FAILED: ${COMPOSE_FILE}${NC}"
fi
echo "================================================================================"

exit ${return_code}
