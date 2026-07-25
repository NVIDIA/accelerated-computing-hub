#!/usr/bin/env python3
"""Restart the current container's persistent Docker Compose siblings."""

import http.client
import json
import os
import socket
import sys
import urllib.parse


DOCKER_SOCKET = "/var/run/docker.sock"
NON_PERSISTENT_SERVICES = {"base"}


class UnixHTTPConnection(http.client.HTTPConnection):
    """An HTTP connection transported over a Unix-domain socket."""

    def __init__(self, socket_path):
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


def docker_request(method, path):
    """Send one request to the Docker Engine API and return its body."""
    connection = UnixHTTPConnection(DOCKER_SOCKET)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()

    if response.status >= 300:
        detail = body.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Docker API {method} {path} returned {response.status}: {detail}"
        )
    return body


def container_details(container_name):
    """Return Docker metadata for a container name or ID."""
    container_path = urllib.parse.quote(container_name, safe="")
    return json.loads(docker_request("GET", f"/containers/{container_path}/json"))


def main():
    if not os.path.exists(DOCKER_SOCKET):
        print(f"Docker socket not found at {DOCKER_SOCKET}")
        return 0

    current = container_details(socket.gethostname())
    labels = current.get("Config", {}).get("Labels", {}) or {}
    project = labels.get("com.docker.compose.project")
    current_service = labels.get("com.docker.compose.service")
    if not project or not current_service:
        raise RuntimeError(
            "Current container has no Docker Compose project/service labels"
        )

    filters = json.dumps({"label": [f"com.docker.compose.project={project}"]})
    query = urllib.parse.urlencode({"all": "1", "filters": filters})
    containers = json.loads(docker_request("GET", f"/containers/json?{query}"))

    siblings = []
    for container in containers:
        sibling_labels = container.get("Labels", {}) or {}
        service = sibling_labels.get("com.docker.compose.service")
        one_off = sibling_labels.get("com.docker.compose.oneoff", "False")
        if (
            not service
            or service == current_service
            or service in NON_PERSISTENT_SERVICES
        ):
            continue
        if one_off.lower() == "true":
            continue
        siblings.append((service, container["Id"]))

    failed = False
    for service, container_id in sorted(siblings):
        try:
            docker_request("POST", f"/containers/{container_id}/restart?t=10")
        except (OSError, RuntimeError) as error:
            print(
                f"Error: Could not restart Compose service {project}/{service}: "
                f"{error}",
                file=sys.stderr,
            )
            failed = True
        else:
            print(f"Restarted Compose service {project}/{service}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
