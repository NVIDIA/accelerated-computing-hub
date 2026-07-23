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
        for key in ("privileged", "ulimits", "deploy"):
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


def replace_repo_volume(data, repo_root, notebooks_root=""):
    """Bind the runtime checkout and, optionally, a persistent notebook tree."""
    volume_name = "accelerated-computing-hub"
    bind_mount = f"{repo_root}:/accelerated-computing-hub"
    notebooks_mount = ""
    if notebooks_root:
        notebooks_mount = (
            f"{notebooks_root}:"
            "/accelerated-computing-hub/tutorials/pyhpc/notebooks"
        )
    for service in (data.get("services") or {}).values():
        volumes = service.get("volumes") or []
        service["volumes"] = [
            bind_mount if item == f"{volume_name}:/accelerated-computing-hub" else item
            for item in volumes
        ]
        if notebooks_mount and notebooks_mount not in service["volumes"]:
            service["volumes"].append(notebooks_mount)
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


def use_host_network(data, all_services=False):
    """Avoid rootless veth creation where the host cannot create one."""
    services = data.get("services") or {}
    selected = services.values() if all_services else (services.get("base"),)
    for service in selected:
        if service is None:
            continue
        service["network_mode"] = "host"
        service.pop("ports", None)

    if all_services:
        jupyter = services.get("jupyter")
        if jupyter is not None:
            environment = jupyter.get("environment") or {}
            if isinstance(environment, dict):
                environment = environment.copy()
                # All services share localhost; socat cannot bind ports already
                # owned by the Streamers and Compose DNS is unavailable.
                environment.update(
                    {
                        "ACH_PORT_FORWARDS": "",
                        "JUPYTER_IP": "127.0.0.1",
                    }
                )
                for name in (
                    "JUPYTER_HTTPS_CERT",
                    "JUPYTER_HTTPS_KEY",
                    "JUPYTER_HOST",
                    "NSYS_HTTP_URL",
                    "SELKIES_ENABLE_HTTPS",
                ):
                    if name in os.environ:
                        environment[name] = os.environ[name]
                jupyter["environment"] = environment

        for service_name in ("nsight", "nsys", "ncu"):
            service = services.get(service_name)
            if service is None:
                continue
            environment = service.get("environment") or {}
            if not isinstance(environment, dict):
                continue
            environment = environment.copy()
            environment.update(
                {
                    "HOST_IP": "127.0.0.1",
                    "SELKIES_ADDR": "127.0.0.1",
                    "SELKIES_TURN_HOST": "127.0.0.1",
                }
            )
            for name in (
                "SELKIES_ENABLE_HTTPS",
                "SELKIES_HTTPS_CERT",
                "SELKIES_HTTPS_KEY",
            ):
                if name in os.environ:
                    environment[name] = os.environ[name]
            if service_name == "ncu":
                # Host networking also shares the HTTP port and abstract X
                # socket namespace. Keep NCU separate from NSYS.
                environment.update(
                    {"DISPLAY": ":5", "HTTP_PORT": "8081", "TURN_PORT": "3479"}
                )
            service["environment"] = environment


def prepare(source, destination, repo_root="", bind_repo=False):
    """Write a Podman-compatible copy of source to destination."""
    with open(source, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    strip_podman_incompatible(data)
    use_host_network(
        data,
        all_services=os.environ.get("ACH_PODMAN_HOST_NETWORK") == "1",
    )
    if bind_repo and repo_root:
        replace_repo_volume(
            data,
            repo_root,
            notebooks_root=os.environ.get("ACH_PODMAN_NOTEBOOKS_ROOT", ""),
        )
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
