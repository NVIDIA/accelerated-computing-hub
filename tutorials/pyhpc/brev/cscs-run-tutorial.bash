#!/usr/bin/env bash
# Launch on Daint and connect from the workstation using one authenticated SSH session.

set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: cscs-run-tutorial.bash [OPTIONS]

Run this script on the workstation with the web browser. It opens one
authenticated SSH control connection to Daint, invokes cscs-launch-tutorial.bash
there, waits for the services, and then invokes cscs-connect-tutorial.bash locally.

Workstation options:
  --user USER     Override the username read from the SSH certificate
  --key PATH      CSCS private key (default: ~/.ssh/cscs-key)

Other options are passed to cscs-launch-tutorial.bash, including --repo, --branch,
--account, --state, --partition, --time, and --start-timeout.
EOF
}

discover_cscs_user() {
    local certificate="${1}-cert.pub"
    [ -f "${certificate}" ] || return 1
    local details
    details=$(ssh-keygen -L -f "${certificate}" 2>/dev/null) || return 1
    awk '
        $1 == "Principals:" { principals = 1; next }
        principals && $1 == "Critical" && $2 == "Options:" { exit }
        principals {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            sub(/[[:space:]]+$/, "", line)
            if (line != "") { count++; value = line }
        }
        END { if (count == 1) print value; else exit 1 }
    ' <<< "${details}"
}

if [ "${1:-}" = -h ] || [ "${1:-}" = --help ]; then
    usage
    exit 0
fi

bootstrap_streamed_helpers() {
    local branch=${ACH_BRANCH:-event/2026-07-cscs-summer-school}
    local base_url=${ACH_CSCS_HELPER_BASE_URL:-https://raw.githubusercontent.com/NVIDIA/accelerated-computing-hub/${branch}/tutorials/pyhpc/brev}
    local helper_dir
    helper_dir=$(mktemp -d "${TMPDIR:-/tmp}/ach-cscs-helpers.XXXXXX")

    cleanup_downloads() {
        rm -f "${helper_dir}/cscs-run-tutorial.bash" \
            "${helper_dir}/cscs-launch-tutorial.bash" \
            "${helper_dir}/cscs-connect-tutorial.bash"
        rmdir "${helper_dir}" >/dev/null 2>&1 || true
    }
    trap cleanup_downloads EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    [ -t 0 ] || cat >/dev/null
    local helper
    for helper in cscs-run-tutorial.bash cscs-launch-tutorial.bash \
        cscs-connect-tutorial.bash; do
        curl --fail --location --retry 3 --silent --show-error \
            "${base_url}/${helper}" --output "${helper_dir}/${helper}"
        chmod 700 "${helper_dir}/${helper}"
    done
    if ! { exec 3</dev/tty; } 2>/dev/null; then
        echo "Error: run this command from an interactive terminal." >&2
        exit 1
    fi
    local status=0
    "${helper_dir}/cscs-run-tutorial.bash" "$@" <&3 3<&- || status=$?
    exit "${status}"
}

if [ -z "${BASH_SOURCE[0]:-}" ]; then
    bootstrap_streamed_helpers "$@"
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)
launch_script="${script_dir}/cscs-launch-tutorial.bash"
connect_script="${script_dir}/cscs-connect-tutorial.bash"
for script in "${launch_script}" "${connect_script}"; do
    if [ ! -x "${script}" ]; then
        echo "Error: missing executable sibling script: ${script}" >&2
        exit 1
    fi
done

user=${CSCS_USER:-}
ssh_key=${CSCS_SSH_KEY:-${HOME:?HOME is not set}/.ssh/cscs-key}
ela_host=${CSCS_ELA_HOST:-ela.cscs.ch}
daint_host=${CSCS_DAINT_HOST:-daint.alps.cscs.ch}
launch_args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --user) user=${2:?--user requires a value}; shift 2 ;;
        --key) ssh_key=${2:?--key requires a value}; shift 2 ;;
        *) launch_args+=("$1"); shift ;;
    esac
done
for host in "${ela_host}" "${daint_host}"; do
    case "${host}" in
        ''|-*|*[!A-Za-z0-9._-]*) echo "Error: invalid CSCS hostname: ${host}" >&2; exit 2 ;;
    esac
done
case "${ssh_key}" in
    *$'\n'*|*$'\r'*|*'"'*) echo "Error: invalid CSCS key path." >&2; exit 2 ;;
esac
if [ ! -f "${ssh_key}" ]; then
    echo "Error: CSCS private key not found: ${ssh_key}" >&2
    exit 2
fi
if [ -z "${user}" ]; then
    user=$(discover_cscs_user "${ssh_key}" || true)
fi
case "${user}" in
    ''|-*|*[!A-Za-z0-9._-]*)
        echo "Error: could not read one CSCS username from ${ssh_key}-cert.pub." >&2
        echo "Run 'cscs-key sign --file ${ssh_key}' or use --user." >&2
        exit 2
        ;;
esac

control_dir=$(mktemp -d "${TMPDIR:-/tmp}/ach-cscs-ssh.XXXXXX")
control_path="${control_dir}/master"
launch_output="${control_dir}/launch.out"
auth_log="${control_dir}/auth.log"
ssh_config="${control_dir}/ssh_config"
cat > "${ssh_config}" <<EOF
Host ach-ela
    HostName ${ela_host}
    User ${user}
    IdentityFile "${ssh_key}"
    IdentitiesOnly yes

Host ach-daint
    HostName ${daint_host}
    User ${user}
    IdentityFile "${ssh_key}"
    IdentitiesOnly yes
    ProxyJump ach-ela
EOF
chmod 600 "${ssh_config}"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    ssh -F "${ssh_config}" -S "${control_path}" -O exit ach-daint >/dev/null 2>&1
    rm -f "${launch_output}" "${auth_log}" "${control_path}" "${ssh_config}"
    rmdir "${control_dir}" >/dev/null 2>&1
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "Opening one authenticated connection to Daint through Ela."
open_master() {
    : > "${auth_log}"
    local status=0
    ssh -F "${ssh_config}" -M -S "${control_path}" -fN \
        -o ControlMaster=yes -o ControlPersist=no \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
        ach-daint 2> "${auth_log}" || status=$?
    if [ "${status}" -ne 0 ]; then
        cat "${auth_log}" >&2
    fi
    return "${status}"
}

if ! open_master; then
    if grep -q 'Permission denied (publickey)' "${auth_log}" && \
       command -v cscs-key >/dev/null 2>&1; then
        echo "SSH authentication was rejected; renewing the CSCS certificate once."
        cscs-key sign --file "${ssh_key}"
        rm -f "${control_path}"
        : > "${auth_log}"
        open_master
    else
        echo "Error: could not authenticate to Daint through Ela." >&2
        echo "Run 'cscs-key sign' if the CSCS certificate has expired, then retry." >&2
        exit 1
    fi
fi

upload_command="umask 077; mkdir -p \"\$HOME/.local/share/accelerated-computing-hub\"; cat > \"\$HOME/.local/share/accelerated-computing-hub/cscs-launch-tutorial.bash\"; chmod 700 \"\$HOME/.local/share/accelerated-computing-hub/cscs-launch-tutorial.bash\""
ssh -F "${ssh_config}" -S "${control_path}" ach-daint "${upload_command}" < "${launch_script}"

remote_command="\"\$HOME/.local/share/accelerated-computing-hub/cscs-launch-tutorial.bash\""
for arg in "${launch_args[@]}"; do
    printf -v quoted_arg '%q' "${arg}"
    remote_command+=" ${quoted_arg}"
done

set +e
ssh -F "${ssh_config}" -S "${control_path}" ach-daint "${remote_command}" \
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
echo "The job continues after that shell exits; while connected, stop it with: scancel --full --signal=TERM ${job_id}"
ACH_CSCS_CONTROL_PATH="${control_path}" ACH_CSCS_SSH_CONFIG="${ssh_config}" \
    "${connect_script}" --user "${user}" --key "${ssh_key}" "${node}"
