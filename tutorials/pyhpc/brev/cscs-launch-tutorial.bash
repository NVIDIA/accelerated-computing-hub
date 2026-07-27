#!/usr/bin/env bash
# Launch the PyHPC web services from a Daint login node or run their Slurm job.

set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: cscs-launch-tutorial.bash [OPTIONS]

Run this script on a Daint login node. It prepares the checkout, submits the
web-service job, waits until all three services are ready, prints the job and
node IDs, and exits.

Options:
  --account ACCOUNT      Override the user's primary CSCS project
  --repo PATH            Checkout path (default: $SCRATCH/accelerated-computing-hub)
  --branch BRANCH        Release and new-checkout branch
                         (default: event/2026-07-cscs-summer-school)
  --state PATH           Deployment state directory (default: $SCRATCH/ach-pyhpc-web)
  --partition NAME       Slurm partition (default: normal)
  --time DURATION        Slurm duration (default: 10:00:00)
  --start-timeout SEC    Time to wait for READY (default: 1800)
  -h, --help             Show this help
EOF
}

prepare_checkout() {
    local repo=$1
    local requested_branch=$2

    if ! git check-ref-format --branch "${requested_branch}" >/dev/null 2>&1; then
        echo "Error: invalid branch name: ${requested_branch}" >&2
        return 2
    fi

    if [ ! -e "${repo}" ]; then
        mkdir -p "$(dirname "${repo}")"
        echo "Cloning ${requested_branch} into ${repo}..."
        git clone --branch "${requested_branch}" \
            https://github.com/NVIDIA/accelerated-computing-hub.git "${repo}"
    elif ! git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Error: --repo exists but is not a Git checkout: ${repo}" >&2
        return 1
    else
        local branch
        branch=$(git -C "${repo}" branch --show-current)
        if [ -z "${branch}" ]; then
            echo "Error: the existing checkout has a detached HEAD: ${repo}" >&2
            return 1
        fi

        local status
        if ! status=$(git -C "${repo}" status --porcelain=v1 --untracked-files=no); then
            echo "Error: could not inspect the existing checkout: ${repo}" >&2
            return 1
        fi

        if [ -z "${status}" ]; then
            local remote_ref="refs/remotes/origin/${requested_branch}"
            echo "Updating the clean checkout to ${requested_branch} in ${repo} (current=${branch})..."
            git -C "${repo}" fetch origin \
                "+refs/heads/${requested_branch}:${remote_ref}"
            if git -C "${repo}" show-ref --verify --quiet \
                "refs/heads/${requested_branch}"; then
                if git -C "${repo}" merge-base --is-ancestor \
                    "refs/heads/${requested_branch}" "${remote_ref}"; then
                    git -C "${repo}" switch "${requested_branch}"
                    git -C "${repo}" merge --ff-only "${remote_ref}"
                elif git -C "${repo}" merge-base --is-ancestor \
                    "${remote_ref}" "refs/heads/${requested_branch}"; then
                    git -C "${repo}" switch "${requested_branch}"
                else
                    echo "Error: local ${requested_branch} cannot fast-forward to origin/${requested_branch}." >&2
                    echo "The checkout remains on ${branch}; resolve it manually or use a different --repo path." >&2
                    return 1
                fi
            else
                git -C "${repo}" switch -c "${requested_branch}" "${remote_ref}"
            fi
        else
            echo "Leaving the dirty checkout unchanged (branch=${branch})."
        fi
    fi

    CHECKOUT_BRANCH=$(git -C "${repo}" branch --show-current)
    if [ -z "${CHECKOUT_BRANCH}" ]; then
        echo "Error: the checkout must be on a branch: ${repo}" >&2
        return 1
    fi
}

queue_state() {
    squeue --noheader --jobs "$1" --format='%T' 2>/dev/null | sed -n '1p'
}

stop_job() {
    local job_id=$1
    local state=
    state=$(queue_state "${job_id}" || true)
    case "${state}" in
        PENDING*|CONFIGURING*|REQUEUED*|RESV_DEL_HOLD*)
            scancel "${job_id}" || true
            ;;
        *)
            scancel --full --signal=TERM "${job_id}" || true
            ;;
    esac

    for _ in {1..30}; do
        if state=$(queue_state "${job_id}"); then
            [ -n "${state}" ] || return 0
        fi
        sleep 1
    done
    echo "Job ${job_id} did not stop after TERM; requesting cancellation." >&2
    scancel "${job_id}" || true
    for _ in {1..30}; do
        if state=$(queue_state "${job_id}"); then
            [ -n "${state}" ] || return 0
        fi
        sleep 1
    done
    echo "Warning: job ${job_id} is still visible in squeue after cancellation." >&2
    return 1
}

wait_until_ready() {
    local job_id=$1
    local log=$2
    local timeout=$3
    local deadline=0
    local last_state=

    while true; do
        local ready_line=
        if [ -f "${log}" ]; then
            ready_line=$(grep -m1 '^READY node=' "${log}" || true)
        fi

        local queue_line=
        if ! queue_line=$(squeue --noheader --jobs "${job_id}" \
            --format='%T|%N' 2>/dev/null); then
            queue_line=
        fi
        queue_line=${queue_line%%$'\n'*}
        local state=
        if [ -n "${queue_line}" ]; then
            state=${queue_line%%|*}
        else
            state=$(sacct --noheader --allocations --jobs "${job_id}" \
                --format=State --parsable2 2>/dev/null | head -n1 | cut -d'|' -f1 || true)
        fi

        if [ -n "${state}" ] && [ "${state}" != "${last_state}" ]; then
            echo "Job ${job_id}: ${state}"
            last_state=${state}
        fi
        case "${state}" in
            RUNNING)
                if [ "${deadline}" -eq 0 ]; then
                    deadline=$((SECONDS + timeout))
                fi
                if [ -n "${ready_line}" ]; then
                    local node=${ready_line#READY node=}
                    printf 'CSCS_WEB_NODE=%s\n' "${node}"
                    printf 'CSCS_WEB_LOG=%s\n' "${log}"
                    return 0
                fi
                ;;
            PENDING*|CONFIGURING*|REQUEUED*|RESV_DEL_HOLD*)
                deadline=0
                ;;
            COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*|PREEMPTED*|BOOT_FAIL*|DEADLINE*)
                echo "Error: job ${job_id} entered ${state} before the services became ready." >&2
                [ ! -f "${log}" ] || tail -n 40 "${log}" >&2
                return 1
                ;;
        esac
        if [ "${deadline}" -ne 0 ] && [ "${SECONDS}" -ge "${deadline}" ]; then
            echo "Error: services did not become ready within ${timeout}s after job ${job_id} started." >&2
            echo "Stopping the job so a retry cannot create a duplicate deployment." >&2
            stop_job "${job_id}" || true
            return 1
        fi
        sleep 2
    done
}

ACTIVE_JOB_ID=
cancel_active_job() {
    trap - HUP INT TERM
    echo "Launcher interrupted; stopping job ${ACTIVE_JOB_ID}." >&2
    stop_job "${ACTIVE_JOB_ID}" || true
    exit 130
}

login_main() {
    if [ "${1:-}" = -h ] || [ "${1:-}" = --help ]; then
        usage
        return 0
    fi

    local account=${CSCS_ACCOUNT:-}
    local repo=${ACH_REPO:-${SCRATCH:?SCRATCH is not set}/accelerated-computing-hub}
    local branch=${ACH_BRANCH:-event/2026-07-cscs-summer-school}
    local state_dir=${ACH_STATE:-${SCRATCH}/ach-pyhpc-web}
    local partition=${CSCS_PARTITION:-normal}
    local duration=${CSCS_TIME:-10:00:00}
    local start_timeout=${ACH_START_TIMEOUT:-1800}

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --account) account=${2:?--account requires a value}; shift 2 ;;
            --repo) repo=${2:?--repo requires a value}; shift 2 ;;
            --branch) branch=${2:?--branch requires a value}; shift 2 ;;
            --state) state_dir=${2:?--state requires a value}; shift 2 ;;
            --partition) partition=${2:?--partition requires a value}; shift 2 ;;
            --time) duration=${2:?--time requires a value}; shift 2 ;;
            --start-timeout) start_timeout=${2:?--start-timeout requires a value}; shift 2 ;;
            -h|--help) usage; return 0 ;;
            *) echo "Error: unknown argument: $1" >&2; usage >&2; return 2 ;;
        esac
    done

    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "Error: run the launcher on a Daint login node, not inside a Slurm job." >&2
        return 1
    fi
    case "${start_timeout}" in
        ''|*[!0-9]*) echo "Error: --start-timeout must be an integer." >&2; return 2 ;;
    esac
    if [ -z "${account}" ]; then
        if ! account=$(id -gn "${USER}") || [ -z "${account}" ]; then
            echo "Error: could not discover the primary CSCS project; use --account." >&2
            return 2
        fi
    fi
    echo "Using CSCS Slurm account ${account}."

    prepare_checkout "${repo}" "${branch}"
    echo "Using checkout branch ${CHECKOUT_BRANCH} with release assets from ${branch}."

    mkdir -p "${state_dir}"
    chmod 700 "${state_dir}"

    local batch_script
    batch_script=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)/$(basename "${BASH_SOURCE[0]}")
    if [ ! -x "${batch_script}" ]; then
        echo "Error: run an executable launcher file, not a pipe or process substitution." >&2
        return 1
    fi

    local -a sbatch_args=(
        --parsable
        "--partition=${partition}"
        "--time=${duration}"
        --nodes=1
        --ntasks=1
        --gpus=1
        --signal=B:TERM@60
        --job-name=ach-pyhpc-web
        "--chdir=${repo}"
        "--output=${state_dir}/slurm-%j.log"
        "--export=ALL,ACH_REPO=${repo},ACH_STATE=${state_dir},ACH_RELEASE_BRANCH=${branch}"
    )
    sbatch_args+=(--account="${account}")

    local job_id
    job_id=$(sbatch "${sbatch_args[@]}" "${batch_script}" --batch)
    job_id=${job_id%%;*}
    case "${job_id}" in
        ''|*[!0-9]*) echo "Error: sbatch returned an invalid job ID: ${job_id}" >&2; return 1 ;;
    esac
    local log="${state_dir}/slurm-${job_id}.log"

    printf 'CSCS_WEB_JOB_ID=%s\n' "${job_id}"
    echo "Waiting for JupyterLab and both Nsight Streamers to become ready..."
    ACTIVE_JOB_ID=${job_id}
    trap cancel_active_job HUP INT TERM
    local wait_status=0
    wait_until_ready "${job_id}" "${log}" "${start_timeout}" || wait_status=$?
    trap - HUP INT TERM
    ACTIVE_JOB_ID=
    if [ "${wait_status}" -ne 0 ]; then
        return "${wait_status}"
    fi
    printf 'CSCS_WEB_REPO=%s\n' "${repo}"
}

batch_main() {
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Error: --batch is for the Slurm job started by this launcher." >&2
    exit 1
fi

ACH_REPO=${ACH_REPO:?ACH_REPO is not set}
ACH_STATE=${ACH_STATE:-${SCRATCH:?SCRATCH is not set}/ach-pyhpc-web}
ACH_RELEASE_BRANCH=${ACH_RELEASE_BRANCH:?ACH_RELEASE_BRANCH is not set}
COMPOSE_URL=${ACH_COMPOSE_URL:-https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/generated/${ACH_RELEASE_BRANCH}/tutorials/pyhpc/brev/docker-compose.yml}
PREPARE_URL=${ACH_PREPARE_URL:-https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/${ACH_RELEASE_BRANCH}/brev/prepare-podman-compose.py}

if [ ! -d "${ACH_REPO}/tutorials/pyhpc/notebooks" ]; then
    echo "Error: student checkout has no PyHPC notebooks: ${ACH_REPO}" >&2
    exit 1
fi

RUN_STATE="${ACH_STATE}/${SLURM_JOB_ID}"
mkdir -p "${RUN_STATE}"
chmod 700 "${ACH_STATE}" "${RUN_STATE}"
SERVICE_EVENTS="${RUN_STATE}/service-events"

SOURCE_COMPOSE="${RUN_STATE}/docker-compose.yml"
PODMAN_COMPOSE="${RUN_STATE}/docker-compose.podman.yml"
PREPARE_SCRIPT="${RUN_STATE}/prepare-podman-compose.py"
curl --fail --location --retry 3 --silent --show-error \
    "${COMPOSE_URL}" --output "${SOURCE_COMPOSE}"
curl --fail --location --retry 3 --silent --show-error \
    "${PREPARE_URL}" --output "${PREPARE_SCRIPT}"

PYTHON=$(command -v python3.11 || command -v python3)
VENV="${RUN_STATE}/venv"
COMPOSE="${VENV}/bin/podman-compose"
if [ ! -x "${COMPOSE}" ]; then
    "${PYTHON}" -m venv "${VENV}"
    "${VENV}/bin/pip" install --disable-pip-version-check --quiet \
        'podman-compose==1.6.0'
fi

TURN_USERNAME="turn_$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 16)"
TURN_PASSWORD="$(openssl rand -base64 48 | tr -dc 'a-zA-Z0-9' | head -c 32)"

export ACH_PODMAN_HOST_NETWORK=1
export JUPYTER_HOST=127.0.0.1
export NSYS_HTTP_URL=http://127.0.0.1:8080
export SELKIES_ENABLE_HTTPS=false

"${VENV}/bin/python" "${PREPARE_SCRIPT}" \
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
stop_services() {
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null || true
    fi
    for store in main nsys ncu; do
        clean_store "${store}"
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    pids=()
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_services
    podman unshare rm -rf "${JOB_ROOT}" >/dev/null 2>&1 || true
    exit "${status}"
}
trap cleanup EXIT INT TERM

# Daint's short-lived user systemd manager can terminate rootless Podman
# processes placed in its transient scope. Keep Podman in the Slurm job cgroup.
unset DBUS_SESSION_BUS_ADDRESS

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
        service_pid=""
        terminating=0
        terminate_service() {
            terminating=1
            if [ -n "${service_pid}" ] && kill -0 "${service_pid}" 2>/dev/null; then
                kill -TERM "${service_pid}" 2>/dev/null || true
            fi
        }
        trap terminate_service TERM INT

        "${COMPOSE}" --podman-run-args=--cgroups=disabled \
            -f "${PODMAN_COMPOSE}" \
            run --rm --no-deps -T \
            -e "TURN_USERNAME=${TURN_USERNAME}" -e "TURN_PASSWORD=${TURN_PASSWORD}" \
            "${service}" &
        service_pid=$!
        service_status=0
        wait "${service_pid}" || service_status=$?
        if [ "${terminating}" -eq 1 ] && kill -0 "${service_pid}" 2>/dev/null; then
            wait "${service_pid}" || service_status=$?
        fi
        printf '%s %s\n' "${service}" "${service_status}" >>"${SERVICE_EVENTS}"
        exit "${service_status}"
    ) >>"${log}" 2>&1 &
    local pid=$!
    chmod 600 "${log}"
    pids+=("${pid}")

    local deadline=$((SECONDS + 300))
    while [ "${SECONDS}" -lt "${deadline}" ]; do
        if [ -s "${SERVICE_EVENTS}" ]; then
            echo "Error: a web service exited while waiting for ${service}; see ${RUN_STATE}/*.log" >&2
            return 1
        fi
        if curl --fail --silent \
            --connect-timeout 1 --max-time 2 "${url}" >/dev/null 2>&1; then
            echo "${service} is ready"
            return 0
        fi
        sleep 1
    done

    echo "Error: timed out waiting for ${service}; see ${log}" >&2
    return 1
}

generation=0
ready_announced=0
while true; do
    generation=$((generation + 1))
    pids=()
    : >"${SERVICE_EVENTS}"
    chmod 600 "${SERVICE_EVENTS}"
    for service in jupyter nsys ncu; do
        printf '\n=== Starting service generation %s ===\n' "${generation}" \
            >>"${RUN_STATE}/${service}.log"
    done

    if ! start_service jupyter main http://127.0.0.1:8888/api/status || \
       ! start_service nsys nsys http://127.0.0.1:8080/health || \
       ! start_service ncu ncu http://127.0.0.1:8081/health; then
        echo "Web service generation ${generation} failed during startup; restarting all services." >&2
        stop_services
        sleep 5
        continue
    fi
    if [ -s "${SERVICE_EVENTS}" ]; then
        echo "A web service exited immediately after startup; restarting all services." >&2
        stop_services
        sleep 5
        continue
    fi

    if [ "${ready_announced}" -eq 0 ]; then
        echo "READY node=$(hostname -s)"
        echo "Logs: ${RUN_STATE}/{jupyter,nsys,ncu}.log"
        ready_announced=1
    else
        echo "Web service generation ${generation} is ready."
    fi

    while [ ! -s "${SERVICE_EVENTS}" ]; do
        sleep 1
    done
    read -r exited_service exited_status <"${SERVICE_EVENTS}"
    echo "${exited_service} exited with status ${exited_status}; restarting JupyterLab and both Nsight Streamers." >&2
    stop_services
    sleep 1
done
}

if [ "${1:-}" = --batch ]; then
    shift
    batch_main "$@"
else
    login_main "$@"
fi
