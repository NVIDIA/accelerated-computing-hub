#! /bin/bash
#
# Helper functions for setting up development mounts with container --user flag.
#
# This library provides functions for:
# - Exporting HOST_UID/HOST_GID for the container engine to match host user permissions
# - Creating container volumes (bind-mount to local repo or plain image-backed)
#
# Usage:
#   source ./brev/dev-common.bash
#   setup_dev_env "/path/to/repo"
#   setup_docker_volume "tutorial-name" true   # bind-mount local repo
#   setup_docker_volume "tutorial-name" false  # image content only


# Select the local container engine. Docker remains the default for Brev and CI,
# while ACH_CONTAINER_ENGINE=podman uses rootless Podman where available (CSCS).
setup_container_engine() {
    if [ -n "${ACH_CONTAINER_ENGINE_CMD:-}" ] && [ -n "${ACH_COMPOSE_CMD:-}" ]; then
        return 0
    fi

    local requested_engine="${ACH_CONTAINER_ENGINE:-auto}"

    if [ "${requested_engine}" = "auto" ]; then
        if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
            requested_engine="docker"
        elif command -v podman >/dev/null 2>&1; then
            requested_engine="podman"
        else
            echo "Error: neither Docker nor Podman is available" >&2
            return 1
        fi
    fi

    case "${requested_engine}" in
        docker)
            if ! command -v docker >/dev/null 2>&1; then
                echo "Error: Docker is not installed or not in PATH" >&2
                return 1
            fi
            export ACH_CONTAINER_ENGINE="docker"
            export ACH_CONTAINER_ENGINE_CMD="docker"
            export ACH_COMPOSE_CMD="docker compose"
            ;;
        podman)
            if ! command -v podman >/dev/null 2>&1; then
                echo "Error: Podman is not installed or not in PATH" >&2
                return 1
            fi
            if command -v podman-compose >/dev/null 2>&1; then
                export ACH_COMPOSE_CMD="podman-compose"
            elif podman compose version >/dev/null 2>&1; then
                export ACH_COMPOSE_CMD="podman compose"
            elif command -v docker-compose >/dev/null 2>&1; then
                export DOCKER_HOST="${DOCKER_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/podman/podman.sock}"
                export ACH_COMPOSE_CMD="docker-compose"
            else
                echo "Error: Podman is available, but no Compose frontend was found" >&2
                echo "Install podman-compose or a podman compose provider." >&2
                return 1
            fi
            export ACH_CONTAINER_ENGINE="podman"
            export ACH_CONTAINER_ENGINE_CMD="podman"
            ;;
        *)
            echo "Error: unsupported ACH_CONTAINER_ENGINE='${requested_engine}' (expected auto, docker, or podman)" >&2
            return 1
            ;;
    esac

    echo "🔧 Container engine: ${ACH_CONTAINER_ENGINE} (${ACH_COMPOSE_CMD})" >&2
}

container() {
    setup_container_engine
    ${ACH_CONTAINER_ENGINE_CMD} "$@"
}

compose() {
    setup_container_engine
    if [ "${ACH_CONTAINER_ENGINE}" = "docker" ]; then
        BUILDKIT_PROGRESS=plain ${ACH_COMPOSE_CMD} "$@"
    else
        ${ACH_COMPOSE_CMD} "$@"
    fi
}

prepare_compose_file() {
    local compose_file=${1}

    setup_container_engine

    if [ "${ACH_CONTAINER_ENGINE}" != "podman" ]; then
        echo "${compose_file}"
        return 0
    fi

    local output_file
    local script_path
    output_file="${TMPDIR:-/tmp}/$(basename "${compose_file}").podman.$$"
    script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" || return; pwd -P)
    python3 "${script_path}/prepare-podman-compose.py" \
        "${compose_file}" \
        "${output_file}" \
        "${ACH_REPO_ROOT:-}" \
        "${ACH_PODMAN_BIND_REPO:-}"
    echo "${output_file}"
}

# Export host user's UID/GID/username for Docker/Podman Compose
setup_dev_env() {
    local REPO_ROOT=${1}

    echo "🔧 Setting up development environment..."
    export ACH_UID=$(id -u)
    export ACH_GID=$(id -g)
    export ACH_USER=$(id -un)
    export ACH_REPO_ROOT="${REPO_ROOT}"
    echo -e "${GREEN:-}✅ ACH_UID=${ACH_UID}, ACH_GID=${ACH_GID}, ACH_USER=${ACH_USER}${NC:-}"
    echo ""
}

# Set up the container volume for a tutorial.
#
# Always removes any existing volume first for a clean state.
# When MOUNT=true, creates a bind-mount volume pointing to the local repo.
# When MOUNT=false, does nothing (compose will create a plain volume
# populated from the image on first use).
setup_docker_volume() {
    local ACH_TUTORIAL=${1}
    local MOUNT=${2:-true}
    local VOLUME_NAME="${ACH_TUTORIAL}_accelerated-computing-hub"

    setup_container_engine

    # Remove existing volume for clean state
    if ${ACH_CONTAINER_ENGINE_CMD} volume inspect "${VOLUME_NAME}" &>/dev/null; then
        echo "🗑️  Removing existing container volume: ${VOLUME_NAME}"
        ${ACH_CONTAINER_ENGINE_CMD} volume rm "${VOLUME_NAME}" &>/dev/null || true
    fi

    if [ "${MOUNT}" = "true" ]; then
        echo "🔧 Creating container volume (bind mount to local repo)..."
        ${ACH_CONTAINER_ENGINE_CMD} volume create --driver local \
          --opt type=none \
          --opt o=bind \
          --opt device="${ACH_REPO_ROOT}" \
          "${ACH_TUTORIAL}_accelerated-computing-hub" > /dev/null
        echo -e "${GREEN:-}✅ container volume created (local mount)${NC:-}"
    else
        echo "📦 Using image content (no local mount)"
    fi
    echo ""
}

# Kept for backward compatibility — equivalent to setup_docker_volume with mount=true.
create_docker_volume() {
    setup_docker_volume "${1}" true
}
