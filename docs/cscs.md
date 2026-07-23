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

Do this once on the laptop or workstation that runs the browser. Native Linux,
macOS, and Windows under WSL are supported; the three helper scripts require
Bash, OpenSSH, Git, and `curl`.

1. Ensure the CSCS account belongs to a project with Daint access and has
   [multi-factor authentication](https://docs.cscs.ch/access/mfa/) configured.
2. Install [`cscs-key`](https://docs.cscs.ch/access/ssh/#command-line-access).
   Homebrew users can run `brew install eth-cscs/tap/cscs-key`; the same page
   links release binaries for Linux, macOS, and Windows.
3. Create a local key once, then obtain a one-day CSCS-signed certificate. The
   signing command opens the CSCS login and MFA flow in the browser:

   ```bash
   mkdir -p -m 700 "${HOME}/.ssh"
   ssh-keygen -t ed25519 -f "${HOME}/.ssh/cscs-key"
   cscs-key sign
   eval "$(ssh-agent -s)"
   ssh-add -t 1d "${HOME}/.ssh/cscs-key"
   ```

   Run the `ssh-agent`/`ssh-add` lines in the terminal that will run the web
   helper. Repeat them in a later terminal if that terminal does not already
   share an agent; do not regenerate the private key.

4. Add the following aliases to `~/.ssh/config`, replacing `YOUR_CSCS_USERNAME`:

   ```ssh-config
   Host ela
       HostName ela.cscs.ch
       User YOUR_CSCS_USERNAME
       IdentityFile ~/.ssh/cscs-key
       IdentitiesOnly yes

   Host daint
       HostName daint.alps.cscs.ch
       User YOUR_CSCS_USERNAME
       ProxyJump ela
       IdentityFile ~/.ssh/cscs-key
       IdentitiesOnly yes

   Host nid*
       User YOUR_CSCS_USERNAME
       IdentityFile ~/.ssh/cscs-key
       IdentitiesOnly yes
   ```

   Protect the file and test the complete path:

   ```bash
   chmod 600 "${HOME}/.ssh/config"
   ssh daint hostname
   ```

CSCS does not support username/password SSH. If the certificate expires, run
`cscs-key sign` again; the private key does not need to be regenerated. See the
official [CSCS SSH instructions](https://docs.cscs.ch/access/ssh/) for account,
MFA, key-signing, and platform-specific installation details.

## Run JupyterLab and Nsight Streamer for 10 hours

The deployment pulls the tutorial image built by GitHub CI and the published
NVIDIA Streamer images; it never builds an image on Daint. It bind-mounts the
checkout's notebook directory into JupyterLab, Nsight Systems, and Nsight
Compute, so student work is saved under `$SCRATCH` and remains after the job
ends. A separate managed release checkout supplies runtime scripts and is never
used for student work; this lets the launcher preserve an older or modified
student checkout without running stale infrastructure from it.

Download the three web helpers from the event branch onto the workstation:

```bash
mkdir -p "${HOME}/ach-cscs-web"
cd "${HOME}/ach-cscs-web"
BASE_URL="https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/event/2026-07-cscs-summer-school/tutorials/pyhpc/brev"
for SCRIPT in launch-cscs-web.bash connect-cscs-web.bash run-cscs-web.bash; do
  curl --fail --location --remote-name "${BASE_URL}/${SCRIPT}"
  chmod +x "${SCRIPT}"
done
```

### Start and connect in one command

This is the recommended path from the workstation:

```bash
./run-cscs-web.bash --account YOUR_CSCS_ACCOUNT
```

The end-to-end helper authenticates to `daint` before sending any script, then
reuses that one SSH control connection for the launch and compute-node tunnel.
Ela and Daint authentication, MFA, and first-use host confirmation therefore
happen before job submission and are not requested a second time. Loading the
private key into `ssh-agent` above also prevents another passphrase prompt when
the final SSH client authenticates to the compute node. If public-key
authentication is rejected, the helper runs `cscs-key sign` and retries once.
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
`launch-cscs-web.bash` there and run:

```bash
./launch-cscs-web.bash --account YOUR_CSCS_ACCOUNT
```

After it reports a node, run the workstation-side connection helper:

```bash
./connect-cscs-web.bash nidXXXXXX
```

This second script prints the three URLs, opens the five forwards, and leaves
the user in a shell on `nidXXXXXX`. Exiting the shell closes the browser access
but does not stop the Slurm job. Re-run the connection script to reconnect.

All five local ports must be free. Ports 8888, 8080, and 8081 carry HTTPS and
WebSocket signaling; ports 3478 and 3479 carry the two Streamers' WebRTC media
and input over TURN/TCP. Forwarding only the HTTPS ports displays the pages but
does not provide working Streamer desktops.

TURN/TCP through SSH was validated with both Streamers: ICE connected through
the relay, input data channels opened, and video frames continued to decode.
Because the media is TCP inside SSH's TCP connection, packet loss can cause
head-of-line stalls. A stable wired connection is recommended, and keeping only
one Streamer tab active reduces bandwidth.

The web applications have no password. Their HTTPS listeners bind only to
compute-node loopback, and the TURN services require random job credentials.
Access is therefore expected only through the SSH connection.

Find or stop the deployment from the workstation:

```bash
ssh daint squeue --me --name=ach-pyhpc-web \
  --format='%.18i %.9T %.10M %.10L %.20N'
ssh daint scancel --full --signal=TERM JOB_ID
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
