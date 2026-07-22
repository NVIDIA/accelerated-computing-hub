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

## Run JupyterLab and Nsight Streamer for 10 hours

The web deployment uses the image built by GitHub CI; it never builds an image
on Daint. It bind-mounts the checkout into JupyterLab, Nsight Systems, and
Nsight Compute, so notebook changes are saved directly under `ACH_REPO` in
`$SCRATCH` and remain after the job ends.

On a Daint login node, clone the event branch once. If the checkout already
exists, switch to the event branch and fast-forward it before students begin
editing notebooks:

```bash
export ACH_BRANCH="event/2026-07-cscs-summer-school"
export ACH_REPO="${SCRATCH}/accelerated-computing-hub"
if [ -d "${ACH_REPO}/.git" ]; then
  git -C "${ACH_REPO}" fetch origin "${ACH_BRANCH}"
  git -C "${ACH_REPO}" switch "${ACH_BRANCH}"
  git -C "${ACH_REPO}" pull --ff-only
else
  git clone --branch "${ACH_BRANCH}" \
    https://github.com/NVIDIA/accelerated-computing-hub.git "${ACH_REPO}"
fi
```

Submit the 10-hour job from that checkout:

```bash
export CSCS_ACCOUNT="YOUR_ACCOUNT"
export ACH_STATE="${SCRATCH}/ach-pyhpc-web"
mkdir -p -m 700 "${ACH_STATE}"
umask 077

cd "${ACH_REPO}"
JOB_ID=$(sbatch --parsable \
  --account="${CSCS_ACCOUNT}" --partition=normal --time=10:00:00 \
  --nodes=1 --ntasks=1 --gpus=1 --signal=B:TERM@60 \
  --job-name=ach-pyhpc-web \
  --chdir="${ACH_REPO}" --output="${ACH_STATE}/slurm-%j.log" \
  --export=ALL,ACH_REPO="${ACH_REPO}",ACH_STATE="${ACH_STATE}",ACH_BRANCH="${ACH_BRANCH}" \
  tutorials/pyhpc/brev/start-cscs-web.bash)
echo "JOB_ID=${JOB_ID}"
tail -F "${ACH_STATE}/slurm-${JOB_ID}.log"
```

Wait for `READY node=nidXXXXXX`, then copy that node name. `Ctrl-C` only stops
`tail`; it does not stop the job. The first launch takes several minutes while
Podman pulls the published tutorial and NVIDIA Streamer images into node memory.

On the workstation where the browser runs, open an SSH tunnel directly to the
allocated compute node through the two CSCS login hosts:

```bash
export CSCS_USER="YOUR_CSCS_USERNAME"
export NODE="nidXXXXXX"

ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -J "${CSCS_USER}@ela.cscs.ch,${CSCS_USER}@daint.alps.cscs.ch" \
  -L 127.0.0.1:8888:127.0.0.1:8888 \
  -L 127.0.0.1:8080:127.0.0.1:8080 \
  -L 127.0.0.1:3478:127.0.0.1:3478 \
  -L 127.0.0.1:8081:127.0.0.1:8081 \
  -L 127.0.0.1:3479:127.0.0.1:3479 \
  "${CSCS_USER}@${NODE}"
```

Keep that terminal open and visit these URLs. Each uses a job-local self-signed
certificate, so the browser asks for confirmation once per URL.

- JupyterLab: <https://127.0.0.1:8888>
- Nsight Systems: <https://127.0.0.1:8080>
- Nsight Compute: <https://127.0.0.1:8081>

All five forwards are required. Ports 8080 and 8081 carry the HTTPS interface
and WebSocket signaling; ports 3478 and 3479 carry the Streamers' WebRTC media
and input data over TURN/TCP. Forwarding only the three HTTPS ports displays the
pages but does not provide a working desktop.

TURN/TCP works through SSH and has been validated with both Streamers: ICE
connected through the relay, the input data channels opened, and video frames
continued to decode. Because the media is TCP inside the SSH TCP connection,
packet loss can cause head-of-line stalls. A stable wired connection is
recommended, and keeping only one Nsight Streamer tab active reduces bandwidth.
If the tunnel drops, reconnect it and reload the browser tabs; the Slurm job and
saved notebook files continue to exist.

The web applications have no password. The helper binds their HTTPS listeners
to compute-node loopback, creates private TURN credentials and logs, and expects
access only through the tunnel. NVIDIA's TURN helper listens on the node's two
TURN ports but accepts only the random job credentials. Do not change the HTTPS
listeners to `0.0.0.0` or share the private state directory.

Find or stop the deployment from a Daint login node:

```bash
squeue --me --name=ach-pyhpc-web \
  --format='%.18i %.9T %.10M %.10L %.20N'
scancel --batch --signal=TERM "${JOB_ID}"
```

The batch-only `TERM` lets the helper remove its containers and node-local image
stores before the allocation ends. The job also stops automatically after 10
hours, and its cleanup leaves student work under `ACH_REPO` untouched. After
the job has stopped, remove only that job's deployment metadata and log:

```bash
rm -rf "${ACH_STATE:?}/${JOB_ID}" "${ACH_STATE:?}/slurm-${JOB_ID}.log"
```

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
