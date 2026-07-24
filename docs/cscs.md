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
checkout's notebook directory into JupyterLab, Nsight Systems, and Nsight
Compute, so student work is saved under `$SCRATCH` and remains after the job
ends. A separate managed release checkout supplies runtime scripts and is never
used for student work; this lets the launcher preserve an older or modified
student checkout without running stale infrastructure from it.

From the workstation, start and connect with one command:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/event/2026-07-cscs-summer-school/tutorials/pyhpc/brev/cscs-run-tutorial |
  bash -s -- --user YOUR_CSCS_USERNAME --account YOUR_CSCS_ACCOUNT
```

On its first run, the helper checks out the event branch in
`${HOME}/accelerated-computing-hub`. It reuses one Daint SSH connection for the
launch and compute-node tunnel; if the certificate expired, it runs
`cscs-key sign` and retries once.
The Daint-side launcher:

- clones the event branch when `$SCRATCH/accelerated-computing-hub` is absent;
- updates an existing checkout to the event branch only when it starts on
  `main` and is completely clean, including untracked files; and
- leaves any modified or non-`main` checkout unchanged.

It submits the 10-hour job, waits until all three HTTPS services report ready,
prints `CSCS_WEB_JOB_ID` and `CSCS_WEB_NODE`, and exits. The workstation helper
then opens all five forwards and leaves the user in a shell on the allocated
compute node. There is no `tail` process to interrupt.

Keep the compute-node shell open and visit these URLs. Each uses a job-local
self-signed certificate, so the browser asks for confirmation once per URL.

- JupyterLab: <https://127.0.0.1:8888>
- Nsight Systems: <https://127.0.0.1:8080>
- Nsight Compute: <https://127.0.0.1:8081>

### Run the launch and connection separately

To launch on a Daint login node without the end-to-end helper, copy or download
`cscs-launch-tutorial` there and run:

```bash
./cscs-launch-tutorial --account YOUR_CSCS_ACCOUNT
```

After it reports a node, run the workstation-side connection helper:

```bash
cd "${HOME}/accelerated-computing-hub/tutorials/pyhpc/brev"
./cscs-connect-tutorial --user YOUR_CSCS_USERNAME nidXXXXXX
```

This second script prints the three URLs, opens the five forwards, and leaves
the user in a shell on `nidXXXXXX`. Exiting the shell closes the browser access
but does not stop the Slurm job. Re-run the connection script to reconnect.

The selected local Jupyter port and the four fixed Streamer ports must be free.
Ports 8888 (the Jupyter default), 8080, and 8081 carry HTTPS and WebSocket
signaling; ports 3478 and 3479 carry the two Streamers' WebRTC media and input
over TURN/TCP. Forwarding only the HTTPS ports displays the pages but does not
provide working Streamer desktops. If only local port 8888 is already in use,
set `ACH_JUPYTER_LOCAL_PORT` before running either workstation helper; the
helper prints the resulting JupyterLab URL. The helper keeps the four Streamer
ports fixed, and its TURN URLs advertise ports 3478 and 3479 to the browser.

TURN/TCP through SSH was validated with both Streamers: ICE connected through
the relay, input data channels opened, and video frames continued to decode.
Because the media is TCP inside SSH's TCP connection, packet loss can cause
head-of-line stalls. A stable wired connection is recommended, and keeping only
one Streamer tab active reduces bandwidth.

The web applications have no password. Their HTTPS listeners bind only to
compute-node loopback, and the TURN services require random job credentials.
Access is therefore expected only through the SSH connection.

Find or stop the deployment from the compute-node shell or a Daint login shell:

```bash
squeue --me --name=ach-pyhpc-web \
  --format='%.18i %.9T %.10M %.10L %.20N'
scancel --full --signal=TERM JOB_ID
```

The job stops automatically after 10 hours. The full-job `TERM` above gives
the helper time to remove its containers, private TLS material, and
node-local image stores while leaving student work in the checkout untouched.

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
