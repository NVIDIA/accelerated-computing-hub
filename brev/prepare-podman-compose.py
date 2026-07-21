#!/usr/bin/env python3
"""Adapt an ACH Docker Compose file for rootless Podman."""

import glob
import os
import stat
import sys

import yaml


GPU_SERVICES = {"base", "jupyter", "nsight", "nsys", "ncu"}
NVIDIA_CONTROL_DEVICES = (
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
)
NVIDIA_DRIVER_LIBRARIES = (
    ("libcuda", "libcuda.so.1"),
    ("libnvidia-ml", "libnvidia-ml.so.1"),
    ("libnvidia-ptxjitcompiler", "libnvidia-ptxjitcompiler.so.1"),
    ("libnvidia-nvvm", "libnvidia-nvvm.so.4"),
)
NVIDIA_LIBRARY_DIRECTORIES = ("/usr/lib/*-linux-gnu", "/usr/lib64")


def strip_podman_incompatible(value):
    """Remove Compose keys unsupported by the rootless Podman setup."""
    if isinstance(value, dict):
        for key in ("build", "privileged", "ulimits", "deploy"):
            value.pop(key, None)
        volumes = value.get("volumes")
        if isinstance(volumes, list):
            value["volumes"] = [
                mount
                for mount in volumes
                if not (
                    isinstance(mount, str)
                    and mount.split(":", 1)[0] == "/var/run/docker.sock"
                )
            ]
        for child in value.values():
            strip_podman_incompatible(child)
    elif isinstance(value, list):
        for child in value:
            strip_podman_incompatible(child)


def find_nvidia_mounts():
    """Return host NVIDIA files to bind-mount and their library directories."""
    paths = []
    if os.path.isfile("/usr/bin/nvidia-smi"):
        paths.append("/usr/bin/nvidia-smi")

    for directory in NVIDIA_LIBRARY_DIRECTORIES:
        for library, _ in NVIDIA_DRIVER_LIBRARIES:
            paths.extend(glob.glob(f"{directory}/{library}.so.*"))

    files = sorted(
        {path for path in paths if os.path.isfile(path) and not os.path.islink(path)}
    )
    mounts = [f"{path}:{path}:ro" for path in files]
    directories = sorted(
        {os.path.dirname(path) for path in files if path != "/usr/bin/nvidia-smi"}
    )
    return mounts, directories


def find_nvidia_devices():
    """Return allocated GPU devices plus the available control devices."""

    def is_character_device(path):
        try:
            return stat.S_ISCHR(os.stat(path).st_mode)
        except OSError:
            return False

    gpu_devices = sorted(
        path
        for path in glob.glob("/dev/nvidia[0-9]*")
        if path.removeprefix("/dev/nvidia").isdigit() and is_character_device(path)
    )
    if not gpu_devices:
        return []
    return gpu_devices + [
        path for path in NVIDIA_CONTROL_DEVICES if is_character_device(path)
    ]


def replace_repo_volume(data, repo_root):
    """Use the checkout directly when a Podman test requests a bind mount."""
    volume_name = "accelerated-computing-hub"
    bind_mount = f"{repo_root}:/accelerated-computing-hub"
    for service in (data.get("services") or {}).values():
        volumes = service.get("volumes") or []
        service["volumes"] = [
            bind_mount if item == f"{volume_name}:/accelerated-computing-hub" else item
            for item in volumes
        ]
        environment = service.get("environment") or {}
        if isinstance(environment, dict):
            environment.update({"ACH_USER": "root", "ACH_UID": "0", "ACH_GID": "0"})
            service["environment"] = environment

    volumes = data.get("volumes")
    if isinstance(volumes, dict):
        volumes.pop(volume_name, None)


def add_rootless_gpu_access(data):
    """Pass the allocated GPU and host driver libraries through to Podman."""
    devices = find_nvidia_devices()
    if not devices:
        return

    mounts, library_directories = find_nvidia_mounts()
    library_links = " ".join(
        f"{library}:{soname}" for library, soname in NVIDIA_DRIVER_LIBRARIES
    )

    for service_name, service in (data.get("services") or {}).items():
        if service_name not in GPU_SERVICES:
            continue

        service_devices = service.get("devices") or []
        service["devices"] = service_devices + [
            device for device in devices if device not in service_devices
        ]

        volumes = service.get("volumes") or []
        service["volumes"] = volumes + [
            mount for mount in mounts if mount not in volumes
        ]

        environment = service.get("environment") or {}
        if isinstance(environment, dict):
            environment["ACH_ROOTLESS_PODMAN"] = "1"
            environment["ACH_NVIDIA_LIBRARY_LINKS"] = library_links
            if library_directories:
                environment["ACH_NVIDIA_LIB_DIRS"] = ":".join(library_directories)
            service["environment"] = environment


def prepare(source, destination, repo_root="", bind_repo=False):
    """Write a Podman-compatible copy of source to destination."""
    with open(source, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    strip_podman_incompatible(data)
    if bind_repo and repo_root:
        replace_repo_volume(data, repo_root)
    add_rootless_gpu_access(data)

    with open(destination, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)


def main():
    if len(sys.argv) != 5:
        raise SystemExit(f"Usage: {sys.argv[0]} SOURCE DESTINATION REPO_ROOT BIND_REPO")
    source, destination, repo_root, bind_repo = sys.argv[1:]
    prepare(source, destination, repo_root, bind_repo == "1")


if __name__ == "__main__":
    main()
