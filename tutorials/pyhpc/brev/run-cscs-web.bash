#!/usr/bin/env bash
# Launch on Daint and connect from the workstation using one authenticated SSH session.

set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: run-cscs-web.bash --account ACCOUNT [LAUNCH OPTIONS]

Run this script on the workstation with the web browser. It opens one
authenticated SSH control connection to Daint, invokes launch-cscs-web.bash
there, waits for the services, and then invokes connect-cscs-web.bash locally.

Launch options are the options accepted by launch-cscs-web.bash, including
--repo, --branch, --state, --partition, --time, and --start-timeout.
EOF
}

if [ "${1:-}" = -h ] || [ "${1:-}" = --help ]; then
    usage
    exit 0
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)
launch_script="${script_dir}/launch-cscs-web.bash"
connect_script="${script_dir}/connect-cscs-web.bash"
for script in "${launch_script}" "${connect_script}"; do
    if [ ! -x "${script}" ]; then
        echo "Error: missing executable sibling script: ${script}" >&2
        exit 1
    fi
done

daint_host=${ACH_DAINT_HOST:-daint}
case "${daint_host}" in
    ''|-*|*[!A-Za-z0-9._-]*) echo "Error: invalid Daint SSH alias: ${daint_host}" >&2; exit 2 ;;
esac
control_dir=$(mktemp -d "${TMPDIR:-/tmp}/ach-cscs-ssh.XXXXXX")
control_path="${control_dir}/master"
launch_output="${control_dir}/launch.out"
auth_log="${control_dir}/auth.log"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    ssh -S "${control_path}" -O exit "${daint_host}" >/dev/null 2>&1
    rm -f "${launch_output}" "${auth_log}" "${control_path}"
    rmdir "${control_dir}" >/dev/null 2>&1
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "Opening one authenticated connection to ${daint_host}."
open_master() {
    : > "${auth_log}"
    local status=0
    ssh -M -S "${control_path}" -fN \
        -o ControlMaster=yes -o ControlPersist=no \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
        "${daint_host}" 2> "${auth_log}" || status=$?
    if [ "${status}" -ne 0 ]; then
        cat "${auth_log}" >&2
    fi
    return "${status}"
}

if ! open_master; then
    if grep -q 'Permission denied (publickey)' "${auth_log}" && \
       command -v cscs-key >/dev/null 2>&1; then
        echo "SSH authentication was rejected; renewing the CSCS certificate once."
        cscs-key sign
        rm -f "${control_path}"
        : > "${auth_log}"
        open_master
    else
        echo "Error: could not authenticate to ${daint_host}." >&2
        echo "Run 'cscs-key sign' if the CSCS certificate has expired, then retry." >&2
        exit 1
    fi
fi

upload_command="umask 077; mkdir -p \"\$HOME/.local/share/ach-cscs-web\"; cat > \"\$HOME/.local/share/ach-cscs-web/launch-cscs-web.bash\"; chmod 700 \"\$HOME/.local/share/ach-cscs-web/launch-cscs-web.bash\""
ssh -S "${control_path}" "${daint_host}" "${upload_command}" < "${launch_script}"

remote_command="\"\$HOME/.local/share/ach-cscs-web/launch-cscs-web.bash\""
for arg in "$@"; do
    printf -v quoted_arg '%q' "${arg}"
    remote_command+=" ${quoted_arg}"
done

set +e
ssh -S "${control_path}" "${daint_host}" "${remote_command}" \
    2>&1 | tee "${launch_output}"
launch_status=${PIPESTATUS[0]}
set -e
if [ "${launch_status}" -ne 0 ]; then
    exit "${launch_status}"
fi

node=$(sed -n 's/^CSCS_WEB_NODE=//p' "${launch_output}" | tail -n1)
job_id=$(sed -n 's/^CSCS_WEB_JOB_ID=//p' "${launch_output}" | tail -n1)
if [ -z "${node}" ] || [ -z "${job_id}" ]; then
    echo "Error: the Daint launcher did not report a job and node ID." >&2
    exit 1
fi
node_number=${node#nid}
case "${node_number}" in
    ''|*[!0-9]*) echo "Error: invalid node ID from the Daint launcher: ${node}" >&2; exit 1 ;;
esac
if [ "${node_number}" = "${node}" ]; then
    echo "Error: invalid node ID from the Daint launcher: ${node}" >&2
    exit 1
fi
case "${job_id}" in
    ''|*[!0-9]*) echo "Error: invalid job ID from the Daint launcher: ${job_id}" >&2; exit 1 ;;
esac

echo "Job ${job_id} is ready on ${node}. Opening the forwarded compute-node shell."
echo "The job continues after that shell exits; stop it with: ssh ${daint_host} scancel --full --signal=TERM ${job_id}"
ACH_DAINT_CONTROL_PATH="${control_path}" ACH_DAINT_HOST="${daint_host}" \
    "${connect_script}" "${node}"
