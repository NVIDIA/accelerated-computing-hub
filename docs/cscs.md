# CSCS rootless containers

CSCS compute nodes do not provide a Docker daemon. Use rootless Podman for
image builds and CSCS Slurm Container Engine EDF files for runs with `srun`.

On Daint, build PyHPC images for ARM64/SBSA. If the registry image is not
multi-arch, build on a compute node and import the result to squashfs before the
allocation exits. CSCS recommends placing rootless Podman storage on `/dev/shm`
because compute-node local Podman state is temporary.

The GitHub image workflow currently publishes `linux/amd64`. Mirroring the CUDA
base for both architectures makes ARM64 builds possible, but does not make that
final PyHPC image multi-arch; use the Daint build/import flow below.

## Build and import on Daint

From the repository checkout, set the account and workspace paths once:

```bash
export CSCS_ACCOUNT="YOUR_ACCOUNT"
export ACH_ROOT="${SCRATCH}/ach-cscs-pyhpc"
export ACH_REPO="$(git rev-parse --show-toplevel)"
mkdir -p "${ACH_ROOT}/images" "${ACH_ROOT}/edf" "${ACH_ROOT}/logs"
```

Build the PyHPC image on a compute node and import it to a persistent squashfs:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 02:00:00 -N1 -n1 bash -lc '
  set -euo pipefail

  storage_base="/dev/shm/${USER}/pyhpc-podman-${SLURM_JOB_ID}"
  storage_conf="${storage_base}/storage.conf"
  mkdir -p "${storage_base}/runroot" "${storage_base}/root"
  {
    echo "[storage]"
    echo "driver = \"overlay\""
    echo "runroot = \"${storage_base}/runroot\""
    echo "graphroot = \"${storage_base}/root\""
  } > "${storage_conf}"
  export CONTAINERS_STORAGE_CONF="${storage_conf}"

  cd "${ACH_REPO}"
  podman build --pull=newer \
    --build-arg LLVM_PARALLEL_LINK_JOBS=1 \
    --build-arg "LLVM_TARGETS_TO_BUILD=AArch64;NVPTX" \
    -t pyhpc-cscs:test \
    -f tutorials/pyhpc/brev/dockerfile .

  output="${ACH_ROOT}/images/pyhpc-cscs.sqsh"
  partial="${output}.partial-${SLURM_JOB_ID}"
  trap "rm -f \"${partial}\"" EXIT
  rm -f "${partial}"
  if ! enroot import -x mount -o "${partial}" podman://pyhpc-cscs:test; then
    echo "enroot import returned nonzero; validating its output" >&2
  fi
  unsquashfs -ll -strict-errors "${partial}" >/dev/null
  mv -f "${partial}" "${output}"
  trap - EXIT
'
```

Generate an EDF for the imported image:

```bash
cd "${ACH_REPO}"
./brev/generate-cscs-edf.bash --mount \
  --image "${ACH_ROOT}/images/pyhpc-cscs.sqsh" \
  --output "${ACH_ROOT}/edf/pyhpc-cscs.toml" \
  pyhpc
```

The generator writes `PMIX_MCA_gds=hash` and `PMIX_MCA_psec=native`, which are
the PMIx settings used for clean Slurm MPI runs on Daint.

## Run tests

Run the CSCS validation driver from the Daint login node:

```bash
cd "${ACH_REPO}"
CSCS_ACCOUNT="${CSCS_ACCOUNT}" \
CSCS_EDF="${ACH_ROOT}/edf/pyhpc-cscs.toml" \
  tutorials/pyhpc/brev/test-cscs.bash
```

The driver runs:

- package smoke tests through the normal tutorial entrypoint
- the full notebook ladder, including `03__mpi4py`
- direct `nsys` and `ncu` command-line smoke checks

The PyHPC image builds `mpi4py` against MPICH for CSCS. Notebook 03 uses the
repository's `mpi4py_launcher.py` helper so the examples run with local
MPICH/Hydra ranks inside the one-rank notebook container step. The helper
forces Hydra's `fork` launcher, avoiding nested use of CSCS's host `srun`
binary from inside the Ubuntu container.

The CppJIT/Thrust SWE solver defines `NVTX_DISABLE` before including Thrust.
This avoids an intermittent AArch64 clang-repl relocation failure in Thrust's
NVTX annotation symbols on CSCS while preserving the GPU Thrust implementation.

For debugging, the individual commands are below.

Run the package smoke tests:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:20:00 -N1 -n1 \
  --environment="${ACH_ROOT}/edf/pyhpc-cscs.toml" \
  env ACH_RUN_TESTS=1 ACH_TEST_ARGS="test/test_packages.py" \
  /accelerated-computing-hub/brev/entrypoint.bash base
```

Run the MPI package smoke test directly:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:10:00 -N1 -n1 \
  --environment="${ACH_ROOT}/edf/pyhpc-cscs.toml" \
  pytest -q /accelerated-computing-hub/tutorials/pyhpc/test/test_packages.py \
  -k mpi4py -s
```

Run the profiling notebooks:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 01:00:00 -N1 -n1 \
  --environment="${ACH_ROOT}/edf/pyhpc-cscs.toml" \
  env ACH_RUN_TESTS=1 ACH_TEST_ARGS="10 or 11 or 12" \
  /accelerated-computing-hub/brev/entrypoint.bash base
```

Run the profilers directly when debugging notebook profiling failures:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:20:00 -N1 -n1 \
  --environment="${ACH_ROOT}/edf/pyhpc-cscs.toml" bash -lc '
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
