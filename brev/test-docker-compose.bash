#! /bin/bash
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
echo "Testing Docker Compose: ${COMPOSE_FILE}"
echo "================================================================================"
echo ""

# Stop any existing containers first
echo "🛑 Stopping any existing containers..."
docker compose -f "${COMPOSE_FILE}" down &>/dev/null || true
echo ""

# Check for and remove existing volume
VOLUME_NAME="${ACH_TUTORIAL}_accelerated-computing-hub"
if docker volume inspect "${VOLUME_NAME}" &>/dev/null; then
    echo "🗑️  Removing existing volume: ${VOLUME_NAME}"
    if docker volume rm "${VOLUME_NAME}" 2>/dev/null; then
        echo -e "${GREEN}✅ Volume removed successfully${NC}"
    else
        echo -e "${YELLOW}⚠️ Warning: Could not remove volume (may still be in use)${NC}"
    fi
    echo ""
fi

export ACH_RUN_TESTS=1


echo "📦 Starting base container..."
echo ""
echo "================================================================================"
# Run base container in foreground to see its output
if docker compose -f "${COMPOSE_FILE}" up base; then
    echo "================================================================================"
    echo ""
    echo -e "${GREEN}✅ Base container completed successfully${NC}"
    echo ""

    # Now start remaining services in detached mode (exclude base)
    echo "📦 Starting remaining services..."
    echo ""
    docker compose -f "${COMPOSE_FILE}" up -d jupyter nsight
    echo ""
else
    echo "================================================================================"
    echo ""
    echo -e "${RED}❌ Base container failed${NC}"
    echo ""

    # Try to clean up
    echo "🛑 Attempting cleanup..."
    docker compose -f "${COMPOSE_FILE}" down || true
    echo ""

    RETURN_CODE=1
fi

if [ ${RETURN_CODE:-0} -eq 0 ] && docker compose -f "${COMPOSE_FILE}" ps | grep -q "Up\|running"; then
    echo -e "${GREEN}✅ Containers started successfully${NC}"
    echo ""

    # Show container status
    echo "📊 Container status:"
    docker compose -f "${COMPOSE_FILE}" ps
    echo ""

    # Capture and display logs (excluding base, which we already saw)
    echo "📋 Container logs:"
    echo "--------------------------------------------------------------------------------"
    docker compose -f "${COMPOSE_FILE}" logs jupyter nsight
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Test restart functionality
    echo "🔄 Testing service restart..."
    echo ""
    if docker compose -f "${COMPOSE_FILE}" restart; then
        echo ""
        echo -e "${GREEN}✅ Services restarted successfully${NC}"
        echo ""

        # Wait a moment for services to stabilize
        echo "⏳ Waiting for services to stabilize..."
        sleep 3
        echo ""

        # Verify containers are still running after restart
        echo "📊 Container status after restart:"
        docker compose -f "${COMPOSE_FILE}" ps
        echo ""

        # Check if any containers are not in running state
        if docker compose -f "${COMPOSE_FILE}" ps | grep -q "Exit\|Restarting"; then
            echo -e "${RED}⚠️  Warning: Some containers are not running after restart${NC}"
            echo ""

            # Show logs for troubleshooting
            echo "📋 Container logs after restart:"
            echo "--------------------------------------------------------------------------------"
            docker compose -f "${COMPOSE_FILE}" logs jupyter nsight
            echo "--------------------------------------------------------------------------------"
            echo ""

            RESTART_FAILED=1
        else
            echo -e "${GREEN}✅ All containers running healthy after restart${NC}"
            echo ""
            RESTART_FAILED=0
        fi
    else
        echo ""
        echo -e "${RED}❌ Failed to restart services${NC}"
        echo ""

        # Show logs for troubleshooting
        echo "📋 Container logs after failed restart:"
        echo "--------------------------------------------------------------------------------"
        docker compose -f "${COMPOSE_FILE}" logs --tail=50 jupyter nsight
        echo "--------------------------------------------------------------------------------"
        echo ""

        RESTART_FAILED=1
    fi

    # Stop containers
    echo "🛑 Stopping containers..."
    if docker compose -f "${COMPOSE_FILE}" down; then
        echo -e "${GREEN}✅ Containers stopped successfully${NC}"
        echo ""

        if [ ${RESTART_FAILED:-0} -eq 1 ]; then
            RETURN_CODE=1
        else
            RETURN_CODE=0
        fi
    else
        echo -e "${RED}❌ Failed to stop containers${NC}"
        echo ""
        RETURN_CODE=1
    fi
else
    echo ""
    echo -e "${RED}❌ Failed to start containers${NC}"
    echo ""

    # Try to capture any logs that might be available
    echo "📋 Attempting to capture logs from failed startup:"
    echo "--------------------------------------------------------------------------------"
    docker compose -f "${COMPOSE_FILE}" logs jupyter nsight || true
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Try to clean up
    echo "🛑 Attempting cleanup..."
    docker compose -f "${COMPOSE_FILE}" down || true
    echo ""

    RETURN_CODE=1
fi

echo ""
echo "================================================================================"
if [ ${RETURN_CODE} -eq 0 ]; then
    echo -e "${GREEN}✅ TEST PASSED: ${COMPOSE_FILE}${NC}"
else
    echo -e "${RED}❌ TEST FAILED: ${COMPOSE_FILE}${NC}"
fi
echo "================================================================================"

exit ${RETURN_CODE}
