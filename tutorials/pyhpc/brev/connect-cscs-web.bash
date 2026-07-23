#!/usr/bin/env bash
# Open the five CSCS web-service forwards and an interactive compute-node shell.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: connect-cscs-web.bash [--user USER] [--daint HOST] NODE

Run this script on the workstation with the web browser. It opens the five
required SSH forwards and, by default, leaves you in a shell on NODE. Exiting
that shell closes the forwards but does not cancel the Slurm job.
EOF
}

user=${CSCS_USER:-}
daint_host=${ACH_DAINT_HOST:-daint}
jupyter_local_port=${ACH_JUPYTER_LOCAL_PORT:-8888}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --user) user=${2:?--user requires a value}; shift 2 ;;
        --daint) daint_host=${2:?--daint requires a value}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "Error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
        *) break ;;
    esac
done

if [ "$#" -eq 0 ]; then
    echo "Error: NODE is required." >&2
    usage >&2
    exit 2
fi
node=$1
shift
if [ "$#" -ne 0 ]; then
    echo "Error: unexpected argument after NODE: $1" >&2
    usage >&2
    exit 2
fi
case "${node}" in
    ''|-*|*[!A-Za-z0-9._-]*) echo "Error: invalid compute-node name: ${node}" >&2; exit 2 ;;
esac
case "${daint_host}" in
    ''|-*|*[!A-Za-z0-9._-]*) echo "Error: invalid Daint SSH alias: ${daint_host}" >&2; exit 2 ;;
esac
case "${jupyter_local_port}" in
    ''|*[!0-9]*|??????*) echo "Error: ACH_JUPYTER_LOCAL_PORT must be a port number." >&2; exit 2 ;;
esac
if [ "${jupyter_local_port}" -lt 1 ] || [ "${jupyter_local_port}" -gt 65535 ]; then
    echo "Error: ACH_JUPYTER_LOCAL_PORT is outside 1-65535." >&2
    exit 2
fi
case "${jupyter_local_port}" in
    8080|8081|3478|3479)
        echo "Error: ACH_JUPYTER_LOCAL_PORT conflicts with a fixed Streamer port." >&2
        exit 2
        ;;
esac

if [ -z "${user}" ]; then
    ssh_config=$(ssh -G "${daint_host}" 2>/dev/null)
    user=$(awk '
        $1 == "user" && !found { value = $2; found = 1 }
        END { print value }
    ' <<< "${ssh_config}")
fi
case "${user}" in
    ''|-*|*[!A-Za-z0-9._-]*)
        echo "Error: set a valid CSCS_USER or configure User for Host ${daint_host}." >&2
        exit 2
        ;;
esac

ssh_args=(
    -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=6
    -L "127.0.0.1:${jupyter_local_port}:127.0.0.1:8888"
    -L 127.0.0.1:8080:127.0.0.1:8080
    -L 127.0.0.1:3478:127.0.0.1:3478
    -L 127.0.0.1:8081:127.0.0.1:8081
    -L 127.0.0.1:3479:127.0.0.1:3479
)

if [ -n "${ACH_DAINT_CONTROL_PATH:-}" ]; then
    printf -v control_path_q '%q' "${ACH_DAINT_CONTROL_PATH}"
    printf -v daint_host_q '%q' "${daint_host}"
    ssh_args+=(
        -o "ProxyCommand=ssh -S ${control_path_q} -W %h:%p ${daint_host_q}"
    )
else
    ssh_args+=(-J "${daint_host}")
fi

cat <<EOF
The HTTPS services are available while this SSH connection remains open:
  JupyterLab:     https://127.0.0.1:${jupyter_local_port}
  Nsight Systems: https://127.0.0.1:8080
  Nsight Compute: https://127.0.0.1:8081
EOF

target="${user}@${node}"
exec ssh -tt "${ssh_args[@]}" "${target}"
