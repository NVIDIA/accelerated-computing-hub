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
    ${ACH_COMPOSE_CMD} "$@"
}

prepare_compose_file() {
    local compose_file=${1}

    setup_container_engine

    if [ "${ACH_CONTAINER_ENGINE}" != "podman" ]; then
        echo "${compose_file}"
        return 0
    fi

    local output_file="${TMPDIR:-/tmp}/$(basename "${compose_file}").podman.$$"
    python3 - "${compose_file}" "${output_file}" "${ACH_REPO_ROOT:-}" "${ACH_PODMAN_BIND_REPO:-}" <<'PY'
import glob
import os
import sys
import yaml

source, destination, repo_root, bind_repo = sys.argv[1:5]
with open(source, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle)

def strip_podman_incompatible(value):
    if isinstance(value, dict):
        for key in ("privileged", "ulimits", "deploy"):
            value.pop(key, None)
        for child in value.values():
            strip_podman_incompatible(child)
    elif isinstance(value, list):
        for child in value:
            strip_podman_incompatible(child)

strip_podman_incompatible(data)

nvidia_devices = [path for path in (
    "/dev/nvidia0",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
) if os.path.exists(path)]
nvidia_mounts = []
nvidia_lib_dirs = set()
for pattern in (
    "/usr/bin/nvidia-smi",
    "/usr/lib/*-linux-gnu/libcuda.so.*",
    "/usr/lib/*-linux-gnu/libnvidia-ml.so.*",
    "/usr/lib/*-linux-gnu/libnvidia-ptxjitcompiler.so.*",
    "/usr/lib/*-linux-gnu/libnvidia-nvvm.so.*",
    "/usr/lib64/libcuda.so.*",
    "/usr/lib64/libnvidia-ml.so.*",
    "/usr/lib64/libnvidia-ptxjitcompiler.so.*",
    "/usr/lib64/libnvidia-nvvm.so.*",
):
    for path in glob.glob(pattern):
        if os.path.isfile(path) and not os.path.islink(path):
            nvidia_mounts.append(f"{path}:{path}:ro")
            if path != "/usr/bin/nvidia-smi":
                nvidia_lib_dirs.add(os.path.dirname(path))

if bind_repo == "1" and repo_root:
    volume_name = "accelerated-computing-hub"
    bind_mount = f"{repo_root}:/accelerated-computing-hub:U"
    services = data.get("services") or {}
    for service in services.values():
        volumes = service.get("volumes") or []
        service["volumes"] = [bind_mount if item == f"{volume_name}:/accelerated-computing-hub" else item for item in volumes]
        environment = service.get("environment") or {}
        if isinstance(environment, dict):
            environment.setdefault("ACH_RUNTIME_DIR", "/tmp/accelerated-computing-hub-runtime")
            environment["ACH_USER"] = "root"
            environment["ACH_UID"] = "0"
            environment["ACH_GID"] = "0"
            service["environment"] = environment
    volumes = data.get("volumes")
    if isinstance(volumes, dict):
        volumes.pop(volume_name, None)
services = data.get("services") or {}
for service_name, service in services.items():
    if service_name in ("base", "jupyter", "nsys", "ncu") and nvidia_devices:
        devices = service.get("devices") or []
        for device in nvidia_devices:
            if device not in devices:
                devices.append(device)
        service["devices"] = devices

        volumes = service.get("volumes") or []
        for mount in nvidia_mounts:
            if mount not in volumes:
                volumes.append(mount)
        service["volumes"] = volumes

        environment = service.get("environment") or {}
        if isinstance(environment, dict):
            if nvidia_lib_dirs:
                driver_lib_path = ":".join(sorted(nvidia_lib_dirs))
                environment["ACH_NVIDIA_LIB_DIRS"] = driver_lib_path
            environment["ACH_ROOTLESS_PODMAN"] = "1"
            service["environment"] = environment

        command = service.get("command")
        entrypoint = service.get("entrypoint") or []
        service_arg = entrypoint[1] if isinstance(entrypoint, list) and len(entrypoint) > 1 else service_name
        service["entrypoint"] = ["/accelerated-computing-hub/brev/entrypoint-podman-gpu.bash", service_arg]
        if command is not None:
            service["command"] = command
        else:
            service.pop("command", None)
with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
PY
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
