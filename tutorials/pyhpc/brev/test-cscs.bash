#! /bin/bash
#
# Run the PyHPC tutorial validation suite on CSCS Slurm Container Engine.
#
# This script is intended to run on a CSCS login node using the committed EDF
# or a caller-provided CSCS_EDF override.

set -euo pipefail

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)

if [ -z "${CSCS_ACCOUNT:-}" ]; then
    echo "Error: set CSCS_ACCOUNT to the CSCS project/account for srun." >&2
    exit 2
fi

if [ -z "${CSCS_EDF:-}" ]; then
    CSCS_EDF="${SCRIPT_PATH}/cscs.toml"
fi

if [ ! -f "${CSCS_EDF}" ]; then
    echo "Error: CSCS_EDF does not exist: ${CSCS_EDF}" >&2
    exit 2
fi

CSCS_PARTITION="${CSCS_PARTITION:-normal}"
CSCS_NOTEBOOK_TIME="${CSCS_NOTEBOOK_TIME:-02:00:00}"
CSCS_PACKAGE_TIME="${CSCS_PACKAGE_TIME:-00:20:00}"
CSCS_PROFILE_TIME="${CSCS_PROFILE_TIME:-00:20:00}"

run_step() {
    local name=$1
    shift

    echo ""
    echo "=========================================="
    echo "${name}"
    echo "=========================================="
    "$@"
}

run_container_tests() {
    local time_limit=$1
    local args=$2

    srun -A "${CSCS_ACCOUNT}" -p "${CSCS_PARTITION}" -t "${time_limit}" -N1 -n1 \
        --environment="${CSCS_EDF}" \
        env ACH_RUN_TESTS=1 ACH_TEST_ARGS="${args}" \
        /accelerated-computing-hub/brev/entrypoint.bash base
}

run_step "Package smoke tests" \
    run_container_tests "${CSCS_PACKAGE_TIME}" "test/test_packages.py"

run_step "Notebook ladder" \
    run_container_tests "${CSCS_NOTEBOOK_TIME}" "test/test_notebooks.py"

# Expand the profiling script inside the allocated container, not here.
# shellcheck disable=SC2016
run_step "Manual nsys/ncu profiler validation" \
    srun -A "${CSCS_ACCOUNT}" -p "${CSCS_PARTITION}" -t "${CSCS_PROFILE_TIME}" -N1 -n1 \
        --environment="${CSCS_EDF}" bash -lc '
            set -euo pipefail
            workdir=$(mktemp -d /tmp/pyhpc-profile.XXXXXX)
            cd "${workdir}"
            cat > profile_smoke.py << "PY"
import cupy as cp
x = cp.arange(1 << 20, dtype=cp.float32)
y = cp.sin(x) + cp.cos(x)
print(float(y.sum()))
cp.cuda.runtime.deviceSynchronize()
PY
            python profile_smoke.py
            nsys profile --stats=false --cuda-event-trace=false \
                --force-overwrite true -o profile_smoke python profile_smoke.py
            test -s profile_smoke.nsys-rep
            nsys export --type sqlite --quiet true --force-overwrite true \
                -o profile_smoke.sqlite profile_smoke.nsys-rep
            test -s profile_smoke.sqlite
            ncu -f --kernel-name regex:.* --set full \
                -o profile_smoke python profile_smoke.py
            test -s profile_smoke.ncu-rep
            ncu --import profile_smoke.ncu-rep --csv | sed -n "1,20p"
        '

echo ""
echo "CSCS PyHPC validation completed successfully."
