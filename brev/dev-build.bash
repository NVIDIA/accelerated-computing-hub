#! /bin/bash

# This script builds Docker containers for tutorials.
#
# Usage:
#   ./dev-build.bash <tutorial-name>
#
# If a tutorial name is provided (e.g., "accelerated-python"), only that tutorial is built.
# If no argument is provided, all tutorials are built.

set -eu

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)

# Function to build a single tutorial
build_tutorial() {
    local TUTORIAL_PATH=${1}
    local TUTORIAL_NAME=$(basename ${TUTORIAL_PATH})

    echo "========================================"
    echo "Building tutorial image: ${TUTORIAL_NAME}"
    echo "========================================"

    if [ ! -f "${TUTORIAL_PATH}/brev/docker-compose.yml" ]; then
        echo "Warning: No docker-compose.yml found at ${TUTORIAL_PATH}/brev/docker-compose.yml"
        echo "Skipping..."
        return 0
    fi

    # Check for HPCCM recipe and generate Dockerfile if needed
    if [ -f "${TUTORIAL_PATH}/brev/docker-recipe.py" ]; then
        echo "Found HPCCM recipe, generating Dockerfile..."
        if ! command -v hpccm &> /dev/null; then
            echo "Error: hpccm not found. Please install it with: pip install hpccm"
            exit 1
        fi
        hpccm --recipe "${TUTORIAL_PATH}/brev/docker-recipe.py" --format docker > "${TUTORIAL_PATH}/brev/dockerfile"
        echo "Dockerfile generated successfully"
    fi

    docker compose --progress=plain -f "${TUTORIAL_PATH}/brev/docker-compose.yml" build

    echo "Successfully built image for ${TUTORIAL_NAME}"
    echo ""
}

# Main logic
if [ $# -eq 0 ]; then
    # No arguments - build all tutorials
    echo "No tutorial specified. Building all tutorials..."
    echo ""

    TUTORIALS=$(${SCRIPT_PATH}/discover-tutorials.bash)

    for TUTORIAL_PATH in ${TUTORIALS}; do
        build_tutorial "${TUTORIAL_PATH}"
    done

    echo "========================================"
    echo "All tutorial images built successfully!"
    echo "========================================"
else
    # Argument provided - build specific tutorial
    TUTORIAL_NAME=${1}
    TUTORIAL_PATH="${REPO_ROOT}/tutorials/${TUTORIAL_NAME}"

    if [ ! -d "${TUTORIAL_PATH}" ]; then
        echo "Error: Tutorial directory not found: ${TUTORIAL_PATH}"
        exit 1
    fi

    build_tutorial "${TUTORIAL_PATH}"
fi
