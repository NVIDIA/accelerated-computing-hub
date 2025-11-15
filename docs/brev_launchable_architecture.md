# Accelerated Computing Hub Brev Launchable Architecture

Brev uses [Docker Compose](https://docs.docker.com/compose/) files to define the services and volumes that define a Launchable.

## Tutorials and Syllabi

A tutorial is a Docker image plus a collection of notebooks, slides and other content that teach a broad subject. A tutorial is defined by a directory: `tutorials/<tutorial-name>`.

Tutorials may contain syllabi. Syllabi are subset of a tutorial's notebooks for a particular learning objective. Each syllabi uses the underlying tutorial's Docker image. A syllabi is defined by a Jupyter notebook file: `tutorials/<tutorial-name>/notebooks/syllabi/<syllabus-name>.ipynb`.

## Docker Compose Files

Each tutorial defines its own Docker Compose file in `tutorials/<tutorial-name>/brev/docker-compose.yml`.

For each syllabi, a Docker Compose file is automatically generated on the [`generated` branch](https://github.com/NVIDIA/accelerated-computing-hub/tree/generated) in `<source-branch>/tutorials/<tutorial-name>/notebooks/syllabi/<syllabus-name>__docker_compose.yml`.

## Docker Compose Services

| Service   | Docker Image | Description |
|-----------|--------------|-------------|
| `base`    | Tutorial     | Performs one-time initialization tasks when a Launchable is deployed, such as updating the Git repository to the latest commit and populating the Docker volume. |
| `jupyter` | Tutorial     | Runs the JupyterLab server and executes notebook content. |
| `nsight`  | [NVIDIA NSight Streamer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-nsys) | Runs the WebRTC server that provides devtools. |

## Docker Images

- Tutorial: A tutorial-specific image built and published by the ACH CI. The image is defined by `tutorials/<tutorial-name>/brev/dockerfile` or `tutorials/<tutorial-name>/brev/docker-recipe.py`.
- [NVIDIA NSight Streamer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-nsys): A pre-built image that serves NSight GUIs over WebRTC.

## Docker Volumes

- `/accelerated-computing-hub`: A Git checkout of the ACH repository mounted by all services.

## Docker Ports

| Service   | Port | Protocol | Description                        |
|-----------|------|----------|------------------------------------|
| `jupyter` | 8888 | HTTP     | JupyterLab.                        |
| `nsight`  | 8080 | HTTP     | WebRTC UI for NSight Streamer.     |
| `nsight`  | 3478 | TURN     | WebRTC stream for NSight Streamer. |
