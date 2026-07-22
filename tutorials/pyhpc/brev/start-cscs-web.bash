#!/usr/bin/env bash
# Start the PyHPC JupyterLab and Nsight Streamer services on a Daint compute node.

set -euo pipefail
umask 077

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Error: submit this script with sbatch; it must run inside a Slurm job." >&2
    exit 1
fi

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)
DEFAULT_REPO=$(cd "${SCRIPT_PATH}/../../.."; pwd -P)
ACH_REPO=${ACH_REPO:-${DEFAULT_REPO}}
ACH_STATE=${ACH_STATE:-${SCRATCH:?SCRATCH is not set}/ach-pyhpc-web}
ACH_BRANCH=${ACH_BRANCH:-event/2026-07-cscs-summer-school}
COMPOSE_URL=${ACH_COMPOSE_URL:-https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/generated/${ACH_BRANCH}/tutorials/pyhpc/brev/docker-compose.yml}

if [ ! -f "${ACH_REPO}/brev/prepare-podman-compose.py" ]; then
    echo "Error: ACH_REPO is not an accelerated-computing-hub checkout: ${ACH_REPO}" >&2
    exit 1
fi

CURRENT_BRANCH=$(git -C "${ACH_REPO}" branch --show-current)
if [ "${CURRENT_BRANCH}" != "${ACH_BRANCH}" ]; then
    echo "Error: ACH_REPO is on '${CURRENT_BRANCH}', expected '${ACH_BRANCH}'." >&2
    exit 1
fi

RUN_STATE="${ACH_STATE}/${SLURM_JOB_ID}"
mkdir -p "${RUN_STATE}" "${ACH_REPO}/logs"
chmod 700 "${ACH_STATE}" "${RUN_STATE}"

SOURCE_COMPOSE="${RUN_STATE}/docker-compose.yml"
PODMAN_COMPOSE="${RUN_STATE}/docker-compose.podman.yml"
curl --fail --location --retry 3 --silent --show-error \
    "${COMPOSE_URL}" --output "${SOURCE_COMPOSE}"

PYTHON=$(command -v python3.11 || command -v python3)
VENV="${RUN_STATE}/venv"
COMPOSE="${VENV}/bin/podman-compose"
if [ ! -x "${COMPOSE}" ]; then
    "${PYTHON}" -m venv "${VENV}"
    "${VENV}/bin/pip" install --disable-pip-version-check --quiet \
        'podman-compose==1.6.0'
fi

TLS_HOST_DIR="${RUN_STATE}/tls"
TLS_CONTAINER_DIR="/run/ach-tls"
rm -rf "${TLS_HOST_DIR}"
mkdir -m 700 "${TLS_HOST_DIR}"
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 1 \
    -keyout "${TLS_HOST_DIR}/localhost.key" \
    -out "${TLS_HOST_DIR}/localhost.crt" \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
    >/dev/null 2>&1
chmod 600 "${TLS_HOST_DIR}/localhost.key" "${TLS_HOST_DIR}/localhost.crt"

TURN_USERNAME="turn_$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 16)"
TURN_PASSWORD="$(openssl rand -base64 48 | tr -dc 'a-zA-Z0-9' | head -c 32)"

export ACH_PODMAN_HOST_NETWORK=1
export JUPYTER_HTTPS_CERT="${TLS_CONTAINER_DIR}/localhost.crt"
export JUPYTER_HTTPS_KEY="${TLS_CONTAINER_DIR}/localhost.key"
export JUPYTER_HOST=127.0.0.1
export NSYS_HTTP_URL=https://127.0.0.1:8080
export SELKIES_ENABLE_HTTPS=true
export SELKIES_HTTPS_CERT="${TLS_CONTAINER_DIR}/localhost.crt"
export SELKIES_HTTPS_KEY="${TLS_CONTAINER_DIR}/localhost.key"

"${VENV}/bin/python" "${ACH_REPO}/brev/prepare-podman-compose.py" \
    "${SOURCE_COMPOSE}" "${PODMAN_COMPOSE}" "${ACH_REPO}" 1

JOB_ROOT="/dev/shm/${USER}/ach-pyhpc-web-${SLURM_JOB_ID}"
mkdir -p "${JOB_ROOT}"
chmod 700 "${JOB_ROOT}"

store_conf() {
    echo "${RUN_STATE}/storage-${1}.conf"
}

store_root() {
    echo "${JOB_ROOT}/${1}"
}

for store in main nsys ncu; do
    root=$(store_root "${store}")
    mkdir -p "${root}/runtime"
    chmod 700 "${root}" "${root}/runtime"
    cat > "$(store_conf "${store}")" <<EOF
[storage]
driver = "overlay"
runroot = "${root}/runroot"
graphroot = "${root}/graphroot"
EOF
done

with_store() {
    local store=$1
    shift
    CONTAINERS_STORAGE_CONF=$(store_conf "${store}") \
    XDG_RUNTIME_DIR="$(store_root "${store}")/runtime" \
        "$@"
}

clean_store() {
    local store=$1
    with_store "${store}" podman rm --all --force >/dev/null 2>&1 || true
    with_store "${store}" podman pod rm --all --force >/dev/null 2>&1 || true
    with_store "${store}" podman system migrate >/dev/null 2>&1 || true
}

pids=()
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null
        wait "${pids[@]}" 2>/dev/null
    fi
    for store in main nsys ncu; do
        clean_store "${store}"
    done
    rm -rf "${TLS_HOST_DIR}"
    podman unshare rm -rf "${JOB_ROOT}" >/dev/null 2>&1 || true
    exit "${status}"
}
trap cleanup EXIT INT TERM

for store in main nsys ncu; do
    clean_store "${store}"
done

echo "Pulling the GitHub CI image and NVIDIA Streamer images (no local builds)..."
with_store main "${COMPOSE}" --podman-run-args=--cgroups=disabled \
    -f "${PODMAN_COMPOSE}" pull jupyter
with_store nsys "${COMPOSE}" --podman-run-args=--cgroups=disabled \
    -f "${PODMAN_COMPOSE}" pull nsys
with_store ncu "${COMPOSE}" --podman-run-args=--cgroups=disabled \
    -f "${PODMAN_COMPOSE}" pull ncu

with_store main "${COMPOSE}" --podman-run-args=--cgroups=disabled \
    -f "${PODMAN_COMPOSE}" \
    run --rm --no-deps -T \
    -e "TURN_USERNAME=${TURN_USERNAME}" -e "TURN_PASSWORD=${TURN_PASSWORD}" \
    base

start_service() {
    local service=$1
    local store=$2
    local url=$3
    local log="${RUN_STATE}/${service}.log"

    (
        export CONTAINERS_STORAGE_CONF
        CONTAINERS_STORAGE_CONF=$(store_conf "${store}")
        export XDG_RUNTIME_DIR
        XDG_RUNTIME_DIR="$(store_root "${store}")/runtime"
        exec "${COMPOSE}" --podman-run-args=--cgroups=disabled \
            -f "${PODMAN_COMPOSE}" \
            run --rm --no-deps -T \
            -e "TURN_USERNAME=${TURN_USERNAME}" -e "TURN_PASSWORD=${TURN_PASSWORD}" \
            -v "${TLS_HOST_DIR}:${TLS_CONTAINER_DIR}:ro" \
            "${service}"
    ) >"${log}" 2>&1 &
    local pid=$!
    chmod 600 "${log}"
    pids+=("${pid}")

    local deadline=$((SECONDS + 300))
    while [ "${SECONDS}" -lt "${deadline}" ]; do
        if curl --fail --insecure --silent \
            --connect-timeout 1 --max-time 2 "${url}" >/dev/null 2>&1; then
            echo "${service} is ready"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "Error: ${service} exited; see ${log}" >&2
            return 1
        fi
        sleep 1
    done

    echo "Error: timed out waiting for ${service}; see ${log}" >&2
    return 1
}

start_service jupyter main https://127.0.0.1:8888/api/status
start_service nsys nsys https://127.0.0.1:8080/health
start_service ncu ncu https://127.0.0.1:8081/health

echo "READY node=$(hostname -s)"
echo "Logs: ${RUN_STATE}/{jupyter,nsys,ncu}.log"

set +e
wait -n "${pids[@]}"
status=$?
set -e
echo "A web service exited; stopping the deployment." >&2
exit "${status}"
