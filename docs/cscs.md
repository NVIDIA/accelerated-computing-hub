# CSCS containers

CSCS compute nodes do not provide a Docker daemon. GitHub Actions builds the
Accelerated Python image natively for `linux/amd64` and `linux/arm64` and
publishes a single multi-architecture image to GHCR. The image includes the
PyHPC course in the shared system Python environment. Daint pulls the ARM64
member through the CSCS Slurm Container Engine; do not build it on Daint.

The published image is public and requires no registry credentials:

```text
ghcr.io/nvidia/accelerated-python-tutorial:main-latest
```

Wait for the main branch's **Build and Push Brev Tutorial Docker Images**
workflow to succeed before starting a Daint run.

## Use the published image on Daint

From the repository checkout, download the branch-specific EDF published by
GitHub CI, then set the account and EDF path:

```bash
export CSCS_ACCOUNT="YOUR_ACCOUNT"
export ACH_REPO="$(git rev-parse --show-toplevel)"
export ACH_BRANCH="main"
export CSCS_EDF="${SCRATCH}/pyhpc-${ACH_BRANCH//\//-}.toml"
curl -fsSL \
  "https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/generated/${ACH_BRANCH}/tutorials/accelerated-python/brev/cscs.toml" \
  --output "${CSCS_EDF}"
```

The generated EDF selects the public published image without bind-mounting a
checkout, so the source and dependencies always come from the same CI build.
Slurm Container Engine selects and caches the ARM64 manifest automatically.

Confirm the published image resolves to ARM64 on Daint:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:10:00 -N1 -n1 -c4 \
  --environment="${CSCS_EDF}" \
  bash -lc 'test "$(dpkg --print-architecture)" = arm64'
```

For a reproducible run, generate a no-mount EDF pinned to the CI commit tag:

```bash
./brev/generate-cscs-edf.bash \
  --image "ghcr.io/nvidia/accelerated-python-tutorial:main-latest" \
  --tag "main-git-<seven-character-git-sha>" \
  --output "${SCRATCH}/pyhpc-cscs-pinned.toml" \
  --workdir "/accelerated-computing-hub/tutorials/accelerated-python/notebooks/pyhpc" \
  accelerated-python
```

Passing both `--image` and `--workdir` avoids a Docker Compose lookup, so this
form also works on Daint login nodes where Docker is unavailable.

The generated EDF sets `PMIX_MCA_gds=hash` and `PMIX_MCA_psec=native`, which
are the PMIx settings used for clean Slurm MPI runs on Daint.

## Set up workstation SSH access

Do this once on the workstation that runs the browser. On x86_64 Linux or WSL,
install [`cscs-key`](https://docs.cscs.ch/access/ssh/) with:

```bash
mkdir -p "${HOME}/.local/bin"
curl -fsSL https://github.com/eth-cscs/cscs-key/releases/download/v1.1.0/cscs-key-v1.1.0-x86_64-unknown-linux-musl.tar.gz |
  tar -xz -C "${HOME}/.local/bin"
export PATH="${HOME}/.local/bin:${PATH}"
```

On macOS, use `brew install eth-cscs/tap/cscs-key`. Then create and sign the
key; `cscs-key sign` opens the CSCS MFA flow in the browser:

```bash
mkdir -p -m 700 "${HOME}/.ssh"
ssh-keygen -t ed25519 -f "${HOME}/.ssh/cscs-key"
cscs-key sign
```

The helpers connect through `ela.cscs.ch` to `daint.alps.cscs.ch` directly; no
SSH aliases are required. Renew an expired certificate with `cscs-key sign`.

## Run JupyterLab and Nsight Streamer for 10 hours

The deployment pulls the tutorial image built by GitHub CI and the published
NVIDIA Streamer images; it never builds an image on Daint. It bind-mounts the
selected checkout into JupyterLab, Nsight Systems, and Nsight Compute, so
student work is saved under `$SCRATCH` and remains after the job ends. Normally
the launcher updates and uses `$SCRATCH/accelerated-computing-hub`. If that
checkout contains work in the former PyHPC location or cannot fast-forward
after the selected branch is rewritten, it preserves the checkout and prepares a
compatible sibling for both the course materials and runtime scripts.

From the workstation, start and connect with one command:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/main/tutorials/accelerated-python/brev/cscs-run-tutorial.bash | bash
```

The helper downloads the three small helper scripts to a temporary directory;
it does not clone the repository on the workstation. It reads the CSCS username
from the signed SSH certificate and uses the user's primary CSCS project as the
Slurm account. It reuses one Daint SSH connection for the launch and
compute-node tunnel; if the certificate expired, it runs `cscs-key sign` and
retries once.
The Daint-side launcher:

- clones the selected branch when `$SCRATCH/accelerated-computing-hub` is absent;
- switches any existing clean checkout to the selected branch and updates it; and
- leaves a checkout with tracked changes unchanged.

If an older checkout cannot fast-forward after the selected branch is rebased, or
still contains the former `tutorials/pyhpc` tree, the launcher preserves it and
uses a sibling checkout whose name includes the selected branch. This protects
tracked, untracked, and ignored exercise output. A compatible sibling is reused;
occupied or obsolete siblings receive a numeric suffix. The launcher prints the
paths so existing student work can be copied deliberately rather than
overwritten or silently hidden.

It submits a 10-hour job with one GPU and 32 CPUs, waits until all three web
services report ready, prints `CSCS_WEB_JOB_ID` and `CSCS_WEB_NODE`, and exits.
The workstation helper then opens all five forwards and leaves the user in a
shell on the allocated compute node. There is no `tail` process to interrupt.
Use `--cpus-per-task` or `CSCS_CPUS_PER_TASK` to change the CPU allocation.

Keep the compute-node shell open and visit these URLs:

- JupyterLab: <http://127.0.0.1:8888>
- Nsight Systems: <http://127.0.0.1:8080>
- Nsight Compute: <http://127.0.0.1:8081>

### Run the launch and connection separately

To launch on a Daint login node without the end-to-end helper, copy or download
`cscs-launch-tutorial.bash` there and run:

```bash
./cscs-launch-tutorial.bash
```

After it reports a node, run the workstation-side connection helper:

```bash
curl -fsSLO https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/main/tutorials/accelerated-python/brev/cscs-connect-tutorial.bash
chmod +x cscs-connect-tutorial.bash
./cscs-connect-tutorial.bash nidXXXXXX
```

This second script prints the three URLs, opens the five forwards, and leaves
the user in a shell on `nidXXXXXX`. Exiting the shell closes the browser access
but does not stop the Slurm job. Re-run the connection script to reconnect.

The selected local Jupyter port and the four fixed Streamer ports must be free.
Ports 8888 (the Jupyter default), 8080, and 8081 carry HTTP and WebSocket
signaling; ports 3478 and 3479 carry the two Streamers' WebRTC media and input
over TURN/TCP. Forwarding only the three HTTP ports displays the pages but does
not provide working Streamer desktops. If only local port 8888 is already in use,
set `ACH_JUPYTER_LOCAL_PORT` before running either workstation helper; the
helper prints the resulting JupyterLab URL. The helper keeps the four Streamer
ports fixed, and its TURN URLs advertise ports 3478 and 3479 to the browser.

TURN/TCP through SSH was validated with both Streamers: ICE connected through
the relay, input data channels opened, and video frames continued to decode.
Because the media is TCP inside SSH's TCP connection, packet loss can cause
head-of-line stalls. A stable wired connection is recommended, and keeping only
one Streamer tab active reduces bandwidth.

The web applications have no password. Their HTTP listeners bind only to
compute-node loopback, and the TURN services require random job credentials.
Access is therefore expected only through the SSH connection.

Find or stop the deployment from the compute-node shell or a Daint login shell:

```bash
squeue --me --name=ach-pyhpc-web \
  --format='%.18i %.9T %.10M %.10L %.20N'
scancel --full --signal=TERM JOB_ID
```

The job stops automatically after 10 hours. The full-job `TERM` above gives
the helper time to remove its containers and node-local image stores while
leaving student work in the checkout untouched.

## Run tests

Run the CSCS validation driver from the Daint login node:

If the web launcher reported a sibling as `CSCS_WEB_REPO`, first set
`ACH_REPO` to that printed path rather than the preserved older checkout.
The driver requests one GPU and 32 CPUs per step. Set `CSCS_CPUS_PER_TASK`,
`CSCS_PARTITION`, or `CSCS_RESERVATION` when the allocation requires different
resources.

```bash
cd "${ACH_REPO}"
CSCS_ACCOUNT="${CSCS_ACCOUNT}" \
CSCS_EDF="${CSCS_EDF}" \
  tutorials/accelerated-python/brev/test-cscs.bash
```

The driver runs:

- package smoke tests through the normal tutorial entrypoint
- the full notebook ladder, including `06__mpi4py`
- direct `nsys` and `ncu` command-line smoke checks

The image has one system Python environment. It selects OpenMPI by default;
the PyHPC and profiler kernels set `IPYTHONDIR=/opt/pyhpc-ipython`,
`MPI4PY_MPIABI=mpich`, and `PATH=/opt/pyhpc-mpi/bin:${PATH}`. Notebook 06
therefore runs local MPICH ranks with the `fork` launcher, avoiding nested use
of the host `srun` launcher.

For debugging, the individual commands are below.

Run the package smoke tests:

```bash
for test_file in test_packages.py test_rapids.py test_pyhpc_packages.py; do
  srun -A "${CSCS_ACCOUNT}" -p normal -t 00:20:00 -N1 -n1 -c32 --gpus=1 \
    --environment="${CSCS_EDF}" \
    env ACH_RUN_TESTS=1 ACH_TEST_ARGS="test/${test_file}" \
    /accelerated-computing-hub/brev/entrypoint.bash base
done
```

Run the MPI package smoke test directly:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:10:00 -N1 -n1 -c4 \
  --environment="${CSCS_EDF}" \
  bash -lc 'MPI4PY_MPIABI=mpich PATH="/opt/pyhpc-mpi/bin:${PATH}" \
    python -m pytest -q \
    /accelerated-computing-hub/tutorials/accelerated-python/test/test_pyhpc_packages.py \
    -k mpi4py -s'
```

Run the profiling notebooks one at a time:

```bash
for notebook in 03 04 05; do
  srun -A "${CSCS_ACCOUNT}" -p normal -t 01:00:00 -N1 -n1 -c32 --gpus=1 \
    --environment="${CSCS_EDF}" \
    env ACH_RUN_TESTS=1 \
    ACH_TEST_ARGS="test/test_pyhpc_notebooks.py -k=${notebook}" \
    /accelerated-computing-hub/brev/entrypoint.bash base
done
```

Run the profilers directly when debugging notebook profiling failures:

```bash
srun -A "${CSCS_ACCOUNT}" -p normal -t 00:20:00 -N1 -n1 -c32 --gpus=1 \
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
