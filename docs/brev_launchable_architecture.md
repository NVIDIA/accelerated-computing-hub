# Accelerated Computing Hub Brev Launchable Architecture

Brev uses [Docker Compose](https://docs.docker.com/compose/) files to define the services and volumes that define a Launchable.

## Tutorial Directory Structure

A tutorial is a Docker image plus a collection of notebooks, slides, and other content that teach a broad subject. Each tutorial is defined by a directory under `tutorials/<tutorial-name>` with the following layout:

```
tutorials/<tutorial-name>/
├── brev/
│   ├── docker-compose.yml     # Docker Compose service definitions
│   ├── dockerfile             # (or docker-recipe.py) Docker image definition
│   ├── requirements.txt       # Python dependencies (if applicable)
│   └── test.bash              # Entrypoint for tutorial-level tests
├── notebooks/
│   ├── <module>/              # Grouped by topic (e.g. fundamentals, kernels, libraries)
│   │   ├── NN__topic.ipynb    # Exercise notebooks (outputs stripped)
│   │   └── solutions/
│   │       └── NN__topic__SOLUTION.ipynb # Solution notebooks (outputs preserved)
│   └── syllabi/               # Subset notebooks for specific learning objectives
├── test/                      # Pytest test modules for this tutorial
└── slides/                    # Presentation materials (optional)
```

## Notebooks

### Naming Convention

Notebook filenames follow the pattern `NN__descriptive_name.ipynb`, where `NN` is a two-digit number and double underscores (`__`) separate the number from the name and between name segments. The number defines the ordering within and across modules; gaps in numbering are expected and make it easy to insert new notebooks later. For example, in the `accelerated-python` tutorial:

```
fundamentals/
  00__what_are_gpus.ipynb
  01__numpy_intro__ndarray_basics.ipynb
  02__numpy_linear_algebra__svd_reconstruction.ipynb
  03__numpy_to_cupy__ndarray_basics.ipynb
  ...
  07__cuda_core__devices_streams_and_memory.ipynb
libraries/
  20__cudf__nyc_parking_violations.ipynb
  21__cudf_pandas__nyc_parking_violations.ipynb
  ...
  28__pynvml.ipynb
kernels/
  40__kernel_authoring__copy.ipynb
  41__kernel_authoring__book_histogram.ipynb
  ...
distributed/
  60__mpi4py.ipynb
  61__dask.ipynb
```

Each module starts at a different tens digit (`00`-`09` for fundamentals, `20`-`29` for libraries, `40`-`49` for kernels, `60`-`69` for distributed), so notebooks have a globally unique ordering across the entire tutorial.

**The notebook number must not appear inside the notebook itself** (not in the title, headings, or body text). For example, notebook `03__numpy_to_cupy__ndarray_basics.ipynb` has the title "NumPy to CuPy - `ndarray` Basics", not "03 - NumPy to CuPy - `ndarray` Basics". This makes renumbering notebooks painless -- only filenames need to change, not content.

### Exercise Notebooks

Exercise notebooks are the files students interact with. They contain instructional content, code examples, and exercises with `TODO` markers for the student to fill in. Exercise notebooks have their **outputs stripped** (no cell outputs, no execution counts) so that diffs remain clean and students see a fresh notebook. The pre-commit `notebook-format` hook enforces this.

### Solution Notebooks

Solution notebooks live in a `solutions/` subdirectory within each module and have `__SOLUTION` appended to the name (e.g. `03__numpy_to_cupy__ndarray_basics__SOLUTION.ipynb`). They contain completed exercises with **outputs preserved** so students (and CI) can see the expected results. CI tests execute every solution notebook end-to-end to verify they run without errors.

### Syllabi

Tutorials may contain syllabi. A syllabus is a subset of a tutorial's notebooks for a particular learning objective. Each syllabus uses the underlying tutorial's Docker image. A syllabus is defined by a Jupyter notebook file: `tutorials/<tutorial-name>/notebooks/syllabi/<syllabus-name>.ipynb`.

## Tests

Each tutorial has a `test/` directory containing [pytest](https://docs.pytest.org/) test modules and a `brev/test.bash` script that serves as the test entrypoint inside Docker. At minimum, tutorials should have a `test_notebooks.py` that discovers and executes all solution notebooks end-to-end. Tutorials may add additional test modules for package validation or other checks as needed.

The `brev/test.bash` script is the single entrypoint that CI calls. It should support running all test suites when invoked with no arguments and forwarding arguments to pytest for targeted runs:

```bash
./test.bash                          # Run all suites
./test.bash 03                       # Run notebook tests matching "03"
./test.bash test/test_notebooks.py   # Run a specific test module
./test.bash -k "cupy"               # Forward raw pytest flags
```

Tests are invoked by CI via `brev/test-docker-compose.bash` (see [CONTRIBUTING.md](../CONTRIBUTING.md) for development script documentation).

## Docker Compose Files

Each tutorial defines its own Docker Compose file in `tutorials/<tutorial-name>/brev/docker-compose.yml`.

For each syllabus, a Docker Compose file is automatically generated on the [`generated` branch](https://github.com/NVIDIA/accelerated-computing-hub/tree/generated) in `<source-branch>/tutorials/<tutorial-name>/notebooks/syllabi/<syllabus-name>__docker_compose.yml`.

## Docker Compose Services

| Service   | Docker Image | Description |
|-----------|--------------|-------------|
| `base`    | Tutorial     | Performs one-time initialization tasks when a Launchable is deployed, such as updating the Git repository to the latest commit and populating the Docker volume. |
| `jupyter` | Tutorial     | Runs the JupyterLab server and executes notebook content. |
| `nsys`    | [NVIDIA Nsight Streamer (nsys)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-nsys) | Runs the WebRTC server for Nsight Systems. |
| `ncu`     | [NVIDIA Nsight Streamer (ncu)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-ncu) | Runs the WebRTC server for Nsight Compute. |

## Docker Images

- Tutorial: A tutorial-specific image built and published by the ACH CI. The image is defined by `tutorials/<tutorial-name>/brev/dockerfile` or `tutorials/<tutorial-name>/brev/docker-recipe.py`.
- [NVIDIA Nsight Streamer (nsys)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-nsys): A pre-built image that serves the Nsight Systems GUI over WebRTC.
- [NVIDIA Nsight Streamer (ncu)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/devtools/containers/nsight-streamer-ncu): A pre-built image that serves the Nsight Compute GUI over WebRTC.

## Docker Volumes

- `/accelerated-computing-hub`: A Git checkout of the ACH repository mounted by all services.

## Docker Ports

| Service   | Port | Protocol | Description                            |
|-----------|------|----------|----------------------------------------|
| `jupyter` | 8888 | HTTP     | JupyterLab.                            |
| `nsys`    | 8080 | HTTP     | WebRTC UI for Nsight Systems Streamer. |
| `nsys`    | 3478 | TURN     | WebRTC stream for Nsight Systems.      |
| `ncu`     | 8081 | HTTP     | WebRTC UI for Nsight Compute Streamer. |
| `ncu`     | 3479 | TURN     | WebRTC stream for Nsight Compute.      |
