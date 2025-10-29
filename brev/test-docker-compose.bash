#! /bin/bash

# This script tests a single Docker Compose file by starting and stopping containers.
#
# Usage:
#   ./test-docker-compose.bash <tutorial-name|docker-compose-file>
#
# Examples:
#   ./test-docker-compose.bash accelerated-python
#   ./test-docker-compose.bash tutorials/accelerated-python/brev/docker-compose.yml

set -euo pipefail

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)

# Check argument
if [ $# -ne 1 ]; then
    echo "Error: Tutorial name or Docker Compose file path is required"
    echo "Usage: $0 <tutorial-name|docker-compose-file>"
    echo "Examples:"
    echo "  $0 accelerated-python"
    echo "  $0 tutorials/accelerated-python/brev/docker-compose.yml"
    exit 1
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
        echo "❌ Error: Tutorial directory not found: ${TUTORIAL_PATH}"
        exit 1
    fi

    COMPOSE_FILE="${TUTORIAL_PATH}/brev/docker-compose.yml"
fi

# Validate docker-compose file exists
if [ ! -f "${COMPOSE_FILE}" ]; then
    echo "❌ Error: Docker Compose file not found: ${COMPOSE_FILE}"
    exit 1
fi

echo "============================================"
echo "Testing Docker Compose: ${COMPOSE_FILE}"
echo "============================================"
echo ""

# Start containers
echo "📦 Starting containers..."
echo ""
if docker compose -f "${COMPOSE_FILE}" up -d; then
    echo ""
    echo "✅ Containers started successfully"
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
    echo "--------------------------------------------"
    docker compose -f "${COMPOSE_FILE}" logs
    echo "--------------------------------------------"
    echo ""

    # Stop containers
    echo "🛑 Stopping containers..."
    if docker compose -f "${COMPOSE_FILE}" down; then
        echo "✅ Containers stopped successfully"
        echo ""
        return_code=0
    else
        echo "❌ Failed to stop containers"
        echo ""
        return_code=1
    fi
else
    echo ""
    echo "❌ Failed to start containers"
    echo ""

    # Try to capture any logs that might be available
    echo "📋 Attempting to capture logs from failed startup:"
    echo "--------------------------------------------"
    docker compose -f "${COMPOSE_FILE}" logs || true
    echo "--------------------------------------------"
    echo ""

    # Try to clean up
    echo "🛑 Attempting cleanup..."
    docker compose -f "${COMPOSE_FILE}" down || true
    echo ""

    return_code=1
fi

echo "============================================"
if [ ${return_code} -eq 0 ]; then
    echo "✅ TEST PASSED: ${COMPOSE_FILE}"
else
    echo "❌ TEST FAILED: ${COMPOSE_FILE}"
fi
echo "============================================"
echo ""

exit ${return_code}
