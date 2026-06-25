# CSCS rootless containers

CSCS systems do not provide a Docker daemon on compute nodes. Use the same
Brev tutorial container images with one of the supported rootless paths:

- `ACH_CONTAINER_ENGINE=podman` for local image builds and Compose-compatible
  smoke tests where Podman and a Compose frontend are available.
- CSCS Slurm Container Engine EDF files for production runs with `srun`.

## Podman development flow

The existing development scripts auto-detect Docker first and then Podman. To
force rootless Podman, set `ACH_CONTAINER_ENGINE=podman`:

```bash
export ACH_CONTAINER_ENGINE=podman
./brev/dev-build.bash pyhpc
./brev/dev-test.bash --no-mount pyhpc
./brev/dev-shell.bash --mount pyhpc base
./brev/dev-stop.bash pyhpc
```

The scripts use `podman-compose`, `podman compose`, or a Docker Compose frontend
pointed at Podman's user socket, depending on what is installed.

Nsight Systems profiling works in rootless Podman without Docker's NVIDIA OCI
hook by bind-mounting the host NVIDIA device nodes and driver libraries. Nsight
Compute metric collection also requires the host NVIDIA driver to allow
non-admin access to GPU performance counters. If NCU reports
`ERR_NVGPUCTRPERM`, ask the site administrator to load the NVIDIA kernel module
with `NVreg_RestrictProfilingToAdminUsers=0` or use the site's documented
profiling-counter access policy.

## CSCS Slurm Container Engine flow

Generate an EDF from the tutorial's committed `docker-compose.yml`:

```bash
./brev/generate-cscs-edf.bash --mount --output "$SCRATCH/ach/pyhpc.toml" pyhpc
```

Run a command inside the image on a CSCS allocation:

```bash
srun --environment="$SCRATCH/ach/pyhpc.toml" \
  /accelerated-computing-hub/brev/entrypoint.bash base
```

To run a tutorial test script inside the container, pass the same environment
used by `brev/dev-test.bash`:

```bash
srun --environment="$SCRATCH/ach/pyhpc.toml" \
  env ACH_RUN_TESTS=1 ACH_TEST_ARGS="" \
  /accelerated-computing-hub/brev/entrypoint.bash base
```

The EDF generator defaults to the image and working directory declared by the
`&image` and `&working-dir` anchors in each tutorial Compose file. Override the
image with `--image` when testing images pushed under a fork namespace.
