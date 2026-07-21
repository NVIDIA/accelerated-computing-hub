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

is_podman() {
    [ "${ACH_CONTAINER_ENGINE}" = "podman" ]
}

setup_test_volume() {
    export ACH_PODMAN_BIND_REPO=0
    if [ "${MOUNT}" = "true" ] && is_podman; then
        export ACH_PODMAN_BIND_REPO=1
        echo "🔧 Using direct Podman bind mount for local repo"
        echo ""
    else
        setup_docker_volume "${ACH_TUTORIAL}" "${MOUNT}"
    fi
}

start_services() {
    local up_args=(up -d)

    # Rootless Podman consumes the image published by GitHub CI; never build it
    # on the target system. Docker development runs may still build with --mount.
    if [ "${MOUNT}" != "true" ] || is_podman; then
        up_args+=(--no-build)
    fi
    if is_podman; then
        up_args+=(base)
    else
        up_args+=(--quiet-pull)
    fi

    compose -f "${COMPOSE_FILE}" "${up_args[@]}"
}

wait_for_services() {
    if ! is_podman; then
        echo "⏳ Waiting for containers to initialize..."
        sleep 5
        return
    fi

    echo "⏳ Waiting for base service tests to finish..."
    BASE_CONTAINER=$(container ps -aq \
        --filter "name=${ACH_TUTORIAL}[-_]base" | head -n 1)
    if [ -z "${BASE_CONTAINER}" ]; then
        echo -e "${RED}❌ Could not find the base service container${NC}"
        return 1
    fi
    if ! container wait --condition=stopped "${BASE_CONTAINER}" >/dev/null; then
        return 1
    fi
    if ! BASE_EXIT_CODE=$(container inspect --format '{{.State.ExitCode}}' \
        "${BASE_CONTAINER}"); then
        return 1
    fi
    echo "Base service exited with code: ${BASE_EXIT_CODE}"
    echo ""
    [ "${BASE_EXIT_CODE}" -eq 0 ]
}

services_started_successfully() {
    [ "${START_FAILED:-1}" -eq 0 ] || return 1

    if is_podman; then
        local state
        local exit_code

        [ "${WAIT_FAILED:-1}" -eq 0 ] || return 1
        [ -n "${BASE_CONTAINER:-}" ] || return 1
        state=$(container inspect --format '{{.State.Status}}' "${BASE_CONTAINER}" 2>/dev/null || true)
        exit_code=$(container inspect --format '{{.State.ExitCode}}' "${BASE_CONTAINER}" 2>/dev/null || true)
        [ "${state}" = "exited" ] && [ "${exit_code}" = "0" ]
    else
        compose -f "${COMPOSE_FILE}" ps | grep -q "Up\|running"
    fi
}

restart_services() {
    if is_podman; then
        echo "🔄 Skipping service restart test for one-shot Podman base service"
        return 0
    fi

    if ! compose -f "${COMPOSE_FILE}" restart; then
        echo ""
        echo -e "${RED}❌ Failed to restart services${NC}"
        echo ""
        echo "📋 Container logs after failed restart:"
        echo "--------------------------------------------------------------------------------"
        compose -f "${COMPOSE_FILE}" logs --tail=50
        echo "--------------------------------------------------------------------------------"
        echo ""
        return 1
    fi

    echo ""
    echo -e "${GREEN}✅ Services restarted successfully${NC}"
    echo ""
    echo "⏳ Waiting for services to stabilize..."
    sleep 5

    for i in 1 2 3; do
        if compose -f "${COMPOSE_FILE}" ps 2>/dev/null | grep -qE "Restarting|restarting"; then
            echo -e "${YELLOW}⚠️  Detected restarting service(s), waiting...${NC}"
            sleep 5
        fi
    done
    echo ""

    echo "📊 Container status after restart:"
    compose -f "${COMPOSE_FILE}" ps
    echo ""
    if compose -f "${COMPOSE_FILE}" ps | grep -qE "Exit|Restarting|restarting"; then
        echo -e "${RED}⚠️  Warning: Some containers are not running after restart${NC}"
        echo ""
        echo "📋 Container logs after restart:"
        echo "--------------------------------------------------------------------------------"
        compose -f "${COMPOSE_FILE}" logs
        echo "--------------------------------------------------------------------------------"
        echo ""
        return 1
    fi

    echo -e "${GREEN}✅ All containers running healthy after restart${NC}"
    echo ""
}

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
setup_test_volume
COMPOSE_FILE=$(prepare_compose_file "${COMPOSE_FILE}")
if [ "${COMPOSE_FILE}" != "${ORIGINAL_COMPOSE_FILE}" ]; then
    echo "Using Podman-compatible Compose file: ${COMPOSE_FILE}"
fi

export ACH_RUN_TESTS=1
export ACH_TEST_ARGS="$*"

# Start container
echo "📦 Starting containers..."
echo ""
if start_services; then
    START_FAILED=0
    echo ""
    echo -e "${GREEN}✅ Containers started successfully${NC}"
    echo ""

    # Wait for containers to initialize, checking for restart loops
    if wait_for_services; then
        WAIT_FAILED=0
    else
        WAIT_FAILED=1
    fi

    # Check multiple times to catch restart loops
    for i in 1 2 3; do
        if compose -f "${COMPOSE_FILE}" ps 2>/dev/null | grep -qE "Restarting|restarting"; then
            echo -e "${YELLOW}⚠️  Detected restarting service(s), waiting...${NC}"
            sleep 5
        fi
    done

    # Final check for restart-looping services
    if compose -f "${COMPOSE_FILE}" ps 2>/dev/null | grep -qE "Restarting|restarting"; then
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
else
    START_FAILED=1
fi

if services_started_successfully; then
    # Show container status
    echo "📊 Container status:"
    compose -f "${COMPOSE_FILE}" ps
    echo ""

    # Capture and display logs
    echo "📋 Container logs:"
    echo "--------------------------------------------------------------------------------"
    compose -f "${COMPOSE_FILE}" logs
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Test restart functionality
    echo "🔄 Testing service restart..."
    echo ""
    if restart_services; then
        RESTART_FAILED=0
    else
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
else
    echo ""
    echo -e "${RED}❌ Failed to start containers${NC}"
    echo ""

    # Try to capture any logs that might be available
    echo "📋 Attempting to capture logs from failed startup:"
    echo "--------------------------------------------------------------------------------"
    compose -f "${COMPOSE_FILE}" logs || true
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
