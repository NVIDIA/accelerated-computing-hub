#!/usr/bin/env python3
"""Restart sibling Docker Compose services through the Docker Engine socket."""

import http.client
import json
import os
import socket
import sys
import urllib.parse


DOCKER_SOCKET = "/var/run/docker.sock"


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path):
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


def docker_request(method, path):
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


def main():
    services = sys.argv[1:]
    if not services:
        print(f"Usage: {sys.argv[0]} <service> [<service> ...]", file=sys.stderr)
        return 2
    if not os.path.exists(DOCKER_SOCKET):
        print(f"Error: Docker socket not found at {DOCKER_SOCKET}", file=sys.stderr)
        return 1

    container_name = socket.gethostname()
    container_path = urllib.parse.quote(container_name, safe="")
    container = json.loads(
        docker_request("GET", f"/containers/{container_path}/json")
    )
    labels = container.get("Config", {}).get("Labels", {}) or {}
    project = labels.get("com.docker.compose.project")
    if not project:
        print(
            "Error: Current container has no com.docker.compose.project label",
            file=sys.stderr,
        )
        return 1

    failed = False
    for service in services:
        filters = json.dumps(
            {
                "label": [
                    f"com.docker.compose.project={project}",
                    f"com.docker.compose.service={service}",
                ]
            }
        )
        query = urllib.parse.urlencode({"all": "1", "filters": filters})
        containers = json.loads(docker_request("GET", f"/containers/json?{query}"))
        if not containers:
            print(
                f"Error: No {service!r} container found in Compose project {project!r}",
                file=sys.stderr,
            )
            failed = True
            continue

        for sibling in containers:
            container_id = sibling["Id"]
            docker_request("POST", f"/containers/{container_id}/restart?t=10")
            print(f"Restarted Compose service {project}/{service}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
