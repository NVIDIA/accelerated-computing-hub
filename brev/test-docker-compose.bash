#! /bin/bash
#
# Test a Docker/Podman Compose file by starting and stopping containers.
#
# This script validates Docker/Podman Compose configurations by attempting to start,
# inspect, and cleanly stop containers.
#
# Usage:
#   ./brev/test-docker-compose.bash [--mount|--no-mount] <tutorial-name|docker-compose-file> [test-args...]
#
# Examples:
#   ./brev/test-docker-compose.bash accelerated-python
#   ./brev/test-docker-compose.bash accelerated-python test/test_notebooks.py -k "01__numpy"
#   ./brev/test-docker-compose.bash --mount accelerated-python
#   ./brev/test-docker-compose.bash tutorials/accelerated-python/brev/docker-compose.yml

set -eu

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)

source "${SCRIPT_PATH}/dev-common.bash"

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") [--mount|--no-mount] <tutorial-name|docker-compose-file> [test-args...]

Test a Docker/Podman Compose file by starting and stopping containers.

Options:
  --mount       Bind-mount local repo into the container
  --no-mount    Use image content only (default)

Arguments:
  tutorial-name          Name of tutorial (e.g., accelerated-python)
  docker-compose-file    Path to docker-compose.yml file
  test-args              Extra arguments forwarded to the tutorial's test.bash

Examples:
  $(basename "$0") accelerated-python
  $(basename "$0") accelerated-python test/test_notebooks.py -k "01__numpy"
  $(basename "$0") --mount accelerated-python
  $(basename "$0") tutorials/accelerated-python/brev/docker-compose.yml

Requirements:
  - Docker Compose or Podman Compose must be installed
EOF
    exit 1
}

# Parse --mount/--no-mount flag (default: no mount)
MOUNT=false
if [ $# -gt 0 ]; then
    case "$1" in
        --mount)    MOUNT=true;  shift ;;
        --no-mount) MOUNT=false; shift ;;
    esac
fi

# Check argument
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Tutorial name or Docker/Podman Compose file path is required${NC}"
    usage
fi

ARG=$1
shift
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
    if [ ! -d "${TUTORIAL_DIR}/brev" ] || [[ ! "${ACH_TUTORIAL}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
        ACH_TUTORIAL=$(awk '/^name:[[:space:]]/ {print $NF; exit}' "${COMPOSE_FILE}")
    fi
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
    echo -e "${RED}Error: Docker/Podman Compose file not found: ${COMPOSE_FILE}${NC}"
    exit 1
fi
ORIGINAL_COMPOSE_FILE="${COMPOSE_FILE}"
echo "================================================================================"
echo "Testing Docker/Podman Compose: ${ORIGINAL_COMPOSE_FILE}"
echo "================================================================================"
echo ""

# Stop any existing containers first
echo "🛑 Stopping any existing containers..."
compose -f "${COMPOSE_FILE}" down &>/dev/null || true
echo ""

# Set up volume (cleanup + optional bind mount)
setup_dev_env "${REPO_ROOT}"
export ACH_PODMAN_BIND_REPO=0
if [ "${MOUNT}" = "true" ] && [ "${ACH_CONTAINER_ENGINE}" = "podman" ]; then
    export ACH_PODMAN_BIND_REPO=1
fi
COMPOSE_FILE=$(prepare_compose_file "${COMPOSE_FILE}")
if [ "${COMPOSE_FILE}" != "${ORIGINAL_COMPOSE_FILE}" ]; then
    echo "Using Podman-compatible Compose file: ${COMPOSE_FILE}"
fi
if [ "${ACH_PODMAN_BIND_REPO}" = "1" ]; then
    echo "🔧 Using direct Podman bind mount for local repo"
    echo ""
else
    setup_docker_volume "${ACH_TUTORIAL}" "${MOUNT}"
fi

export ACH_RUN_TESTS=1
export ACH_TEST_ARGS="$*"

if [ "${MOUNT}" != "true" ]; then
    BASE_IMAGE=$(awk '/image:[[:space:]]*&image[[:space:]]/ {print $3; exit}' "${COMPOSE_FILE}")
    if [ -n "${BASE_IMAGE}" ]; then
        echo "🔎 Checking base image availability: ${BASE_IMAGE}"
        if ! container image exists "${BASE_IMAGE}" && ! container pull "${BASE_IMAGE}"; then
            echo -e "${RED}❌ Base image is not available: ${BASE_IMAGE}${NC}"
            echo "Build it first with: ACH_CONTAINER_ENGINE=${ACH_CONTAINER_ENGINE} ./brev/dev-build.bash ${ACH_TUTORIAL}"
            exit 1
        fi
        echo ""
    fi
fi

# Start container
echo "📦 Starting containers..."
echo ""
UP_ARGS="up -d"
if [ "${ACH_CONTAINER_ENGINE}" = "docker" ]; then
    UP_ARGS="${UP_ARGS} --quiet-pull"
fi
if [ "${MOUNT}" != "true" ]; then
    UP_ARGS="${UP_ARGS} --no-build"
fi
if [ "${ACH_CONTAINER_ENGINE}" = "podman" ]; then
    UP_ARGS="${UP_ARGS} base"
fi

if compose -f "${COMPOSE_FILE}" ${UP_ARGS}; then
    echo ""
    echo -e "${GREEN}✅ Containers started successfully${NC}"
    echo ""

    if [ "${ACH_CONTAINER_ENGINE}" = "podman" ]; then
        echo "⏳ Waiting for base service tests to finish..."
        BASE_CONTAINER="${ACH_TUTORIAL}_base_1"
        ${ACH_CONTAINER_ENGINE_CMD} wait --condition=stopped "${BASE_CONTAINER}" >/dev/null
        BASE_EXIT_CODE=$(${ACH_CONTAINER_ENGINE_CMD} inspect --format '{{.State.ExitCode}}' "${BASE_CONTAINER}")
        echo "Base service exited with code: ${BASE_EXIT_CODE}"
        echo ""
    fi

    # Wait for containers to initialize, checking for restart loops
    if [ "${ACH_CONTAINER_ENGINE}" != "podman" ]; then
        echo "⏳ Waiting for containers to initialize..."
        sleep 5
    fi

    # Check multiple times to catch restart loops
    for i in 1 2 3; do
        if compose -f "${COMPOSE_FILE}" ps 2>/dev/null | grep -qE "Restarting|restarting"; then
            echo -e "${YELLOW}⚠️  Detected restarting service(s), waiting...${NC}"
            sleep 5
        fi
    done

    # Final check for restart-looping services
    if [ "${ACH_CONTAINER_ENGINE}" != "podman" ] && compose -f "${COMPOSE_FILE}" ps 2>/dev/null | grep -qE "Restarting|restarting"; then
        echo -e "${RED}❌ Service(s) stuck in restart loop${NC}"
        echo ""
        echo "📊 Container status:"
        compose -f "${COMPOSE_FILE}" ps
        echo ""
        echo "📋 Container logs:"
        echo "--------------------------------------------------------------------------------"
        compose -f "${COMPOSE_FILE}" logs --tail=100
        echo "--------------------------------------------------------------------------------"
        echo ""

        # Clean up
        echo "🛑 Stopping containers..."
        compose -f "${COMPOSE_FILE}" down || true
        echo ""
        echo "================================================================================"
        echo -e "${RED}❌ TEST FAILED: ${COMPOSE_FILE}${NC}"
        echo "================================================================================"
        exit 1
    fi
    echo ""
fi

compose_ps() {
    if [ "${ACH_CONTAINER_ENGINE}" = "podman" ]; then
        ${ACH_CONTAINER_ENGINE_CMD} ps -a --filter "name=${ACH_TUTORIAL}_" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
    else
        compose -f "${COMPOSE_FILE}" ps
    fi
}

compose_logs() {
    if [ "${ACH_CONTAINER_ENGINE}" = "podman" ]; then
        ${ACH_CONTAINER_ENGINE_CMD} logs "${ACH_TUTORIAL}_base_1" || true
    else
        compose -f "${COMPOSE_FILE}" logs "$@"
    fi
}

compose_started_successfully() {
    if [ "${ACH_CONTAINER_ENGINE}" = "podman" ]; then
        local state
        local exit_code
        state=$(${ACH_CONTAINER_ENGINE_CMD} inspect --format '{{.State.Status}}' "${ACH_TUTORIAL}_base_1" 2>/dev/null || true)
        exit_code=$(${ACH_CONTAINER_ENGINE_CMD} inspect --format '{{.State.ExitCode}}' "${ACH_TUTORIAL}_base_1" 2>/dev/null || true)
        [ "${state}" = "running" ] || { [ "${state}" = "exited" ] && [ "${exit_code}" = "0" ]; }
    else
        compose -f "${COMPOSE_FILE}" ps | grep -q "Up\|running"
    fi
}

if compose_started_successfully; then
    # Show container status
    echo "📊 Container status:"
    compose_ps
    echo ""

    # Capture and display logs
    echo "📋 Container logs:"
    echo "--------------------------------------------------------------------------------"
    compose_logs
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Test restart functionality
    echo "🔄 Testing service restart..."
    echo ""
    if [ "${ACH_CONTAINER_ENGINE}" = "podman" ]; then
        echo "🔄 Skipping service restart test for one-shot Podman base service"
        RESTART_FAILED=0
    elif compose -f "${COMPOSE_FILE}" restart; then
        echo ""
        echo -e "${GREEN}✅ Services restarted successfully${NC}"
        echo ""

        # Wait for services to stabilize, checking for restart loops
        echo "⏳ Waiting for services to stabilize..."
        sleep 5

        # Check multiple times to catch restart loops
        for i in 1 2 3; do
            if compose -f "${COMPOSE_FILE}" ps 2>/dev/null | grep -qE "Restarting|restarting"; then
                echo -e "${YELLOW}⚠️  Detected restarting service(s), waiting...${NC}"
                sleep 5
            fi
        done
        echo ""

        # Verify containers are still running after restart
        echo "📊 Container status after restart:"
        compose -f "${COMPOSE_FILE}" ps
        echo ""

        # Check if any containers are not in running state or stuck restarting
        if compose -f "${COMPOSE_FILE}" ps | grep -qE "Exit|Restarting|restarting"; then
            echo -e "${RED}⚠️  Warning: Some containers are not running after restart${NC}"
            echo ""

            # Show logs for troubleshooting
            echo "📋 Container logs after restart:"
            echo "--------------------------------------------------------------------------------"
            compose_logs
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
        compose_logs --tail=50
        echo "--------------------------------------------------------------------------------"
        echo ""

        RESTART_FAILED=1
    fi

    # Stop containers
    echo "🛑 Stopping containers..."
    if compose -f "${COMPOSE_FILE}" down; then
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

    if [ "${ACH_CONTAINER_ENGINE}" = "podman" ] && [ "${BASE_EXIT_CODE:-0}" -ne 0 ]; then
        RETURN_CODE=1
    fi
else
    echo ""
    echo -e "${RED}❌ Failed to start containers${NC}"
    echo ""

    # Try to capture any logs that might be available
    echo "📋 Attempting to capture logs from failed startup:"
    echo "--------------------------------------------------------------------------------"
    compose_logs || true
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Try to clean up
    echo "🛑 Attempting cleanup..."
    compose -f "${COMPOSE_FILE}" down || true
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
