# CSCS containers

CSCS compute nodes do not provide a Docker daemon. GitHub Actions builds the
PyHPC image natively for `linux/amd64` and `linux/arm64` and publishes a single
multi-architecture image to GHCR. Daint pulls the ARM64 member through the CSCS
Slurm Container Engine; do not build the image on Daint.

The event image is public and requires no registry credentials:

```text
ghcr.io/nvidia/pyhpc-tutorial:event-2026-07-cscs-summer-school-latest
```

Wait for the event branch's **Build and Push Brev Tutorial Docker Images**
workflow to succeed before starting a Daint run.

## Use the published image on Daint

From the repository checkout, download the branch-specific EDF published by
GitHub CI, then set the account and EDF path:

```bash
export CSCS_ACCOUNT="YOUR_ACCOUNT"
export ACH_REPO="$(git rev-parse --show-toplevel)"
export ACH_BRANCH="event/2026-07-cscs-summer-school"
export CSCS_EDF="${SCRATCH}/pyhpc-${ACH_BRANCH//\//-}.toml"
curl -fsSL \
  "https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/generated/${ACH_BRANCH}/tutorials/pyhpc/brev/cscs.toml" \
  --output "${CSCS_EDF}"
```

The generated EDF selects the public event image without bind-mounting a
checkout, so the source and dependencies always come from the same CI build.
Slurm Container Engine selects and caches the ARM64 manifest automatically.

Confirm the published image resolves to ARM64 on Daint:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:10:00 -N1 -n1 \
  --environment="${CSCS_EDF}" \
  bash -lc 'test "$(dpkg --print-architecture)" = arm64'
```

For a reproducible run, generate a no-mount EDF pinned to the CI commit tag:

```bash
./brev/generate-cscs-edf.bash \
  --tag "event-2026-07-cscs-summer-school-git-<seven-character-git-sha>" \
  --output "${SCRATCH}/pyhpc-cscs-pinned.toml" \
  pyhpc
```

The generated EDF sets `PMIX_MCA_gds=hash` and `PMIX_MCA_psec=native`, which
are the PMIx settings used for clean Slurm MPI runs on Daint.

## Run tests

Run the CSCS validation driver from the Daint login node:

```bash
cd "${ACH_REPO}"
CSCS_ACCOUNT="${CSCS_ACCOUNT}" \
CSCS_EDF="${CSCS_EDF}" \
  tutorials/pyhpc/brev/test-cscs.bash
```

The driver runs:

- package smoke tests through the normal tutorial entrypoint
- the full notebook ladder, including `03__mpi4py`
- direct `nsys` and `ncu` command-line smoke checks

The PyHPC image builds `mpi4py` against MPICH. Notebook 03 runs local ranks with
`mpirun.mpich -launcher fork`, avoiding nested use of the host `srun` launcher.

For debugging, the individual commands are below.

Run the package smoke tests:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:20:00 -N1 -n1 \
  --environment="${CSCS_EDF}" \
  env ACH_RUN_TESTS=1 ACH_TEST_ARGS="test/test_packages.py" \
  /accelerated-computing-hub/brev/entrypoint.bash base
```

Run the MPI package smoke test directly:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:10:00 -N1 -n1 \
  --environment="${CSCS_EDF}" \
  pytest -q /accelerated-computing-hub/tutorials/pyhpc/test/test_packages.py \
  -k mpi4py -s
```

Run the profiling notebooks:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 01:00:00 -N1 -n1 \
  --environment="${CSCS_EDF}" \
  env ACH_RUN_TESTS=1 ACH_TEST_ARGS="10 or 11 or 12" \
  /accelerated-computing-hub/brev/entrypoint.bash base
```

Run the profilers directly when debugging notebook profiling failures:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:20:00 -N1 -n1 \
  --environment="${CSCS_EDF}" bash -lc '
    set -euo pipefail
    cd /tmp
    cat > profile_smoke.py << "PY"
import cupy as cp
x = cp.arange(1 << 20, dtype=cp.float32)
y = cp.sin(x) + cp.cos(x)
print(float(y.sum()))
cp.cuda.runtime.deviceSynchronize()
PY
    nsys profile --stats=false --cuda-event-trace=false \
      --force-overwrite true -o profile_smoke python profile_smoke.py
    nsys export --type sqlite --quiet true --force-overwrite true \
      -o profile_smoke.sqlite profile_smoke.nsys-rep
    ncu -f --kernel-name regex:.* --set full \
      -o profile_smoke python profile_smoke.py
    ncu --import profile_smoke.ncu-rep --csv | sed -n "1,20p"
  '
```

Nsight Compute metric collection requires the site driver to allow access to GPU
performance counters. If NCU reports `ERR_NVGPUCTRPERM`, use the CSCS profiling
counter policy for the target system.
