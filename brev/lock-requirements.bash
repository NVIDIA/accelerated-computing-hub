#! /bin/bash
#
# Generate or check a tutorial's brev/requirements.lock.txt from its
# brev/requirements.txt.
#
# The lock pins every package, including dependencies, with hashes, and the
# tutorial's image installs only from it. It is resolved inside an Ubuntu 22.04
# container with the deadsnakes Python 3.12 that the tutorial images use, so
# the host's Python does not matter. Existing pins are kept unless
# requirements.txt no longer allows them or --upgrade is given.
#
# Usage:
#   brev/lock-requirements.bash <tutorial>...            # update the locks
#   brev/lock-requirements.bash --upgrade <tutorial>...  # move to the newest allowed versions
#   brev/lock-requirements.bash --check <tutorial>...    # fail if a lock is out of date
#
# With no tutorials, every tutorial that has a brev/requirements.lock.txt is used.

set -euo pipefail

PYTHON_VERSION=3.12
PIP_VERSION=26.2.2
PIP_TOOLS_VERSION=7.6.2
LOCK_IMAGE=ubuntu:22.04

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_PATH}/.." && pwd)

MODE=update
case "${1:-}" in
    --check) MODE=check; shift ;;
    --upgrade) MODE=upgrade; shift ;;
esac

if [ $# -eq 0 ]; then
    mapfile -t TUTORIALS < <(cd "${REPO_ROOT}" && \
        ls tutorials/*/brev/requirements.lock.txt | cut -d/ -f2)
else
    TUTORIALS=("$@")
fi

UPGRADE_FLAG=""
[ "${MODE}" = upgrade ] && UPGRADE_FLAG="--upgrade"

STATUS=0
for TUTORIAL in "${TUTORIALS[@]}"; do
    BREV_DIR="tutorials/${TUTORIAL}/brev"
    if [ ! -f "${REPO_ROOT}/${BREV_DIR}/requirements.txt" ]; then
        echo "❌ ${BREV_DIR}/requirements.txt not found"
        STATUS=1
        continue
    fi
    echo "Locking ${TUTORIAL}..."

    NEW_LOCK=$(mktemp)
    # The repository is mounted read-only; the lock is written to stdout.
    docker run --rm --network host \
        -v "${REPO_ROOT}:/src:ro" \
        -v ach-lock-pip-cache:/root/.cache/pip \
        -w "/src/${BREV_DIR}" \
        -e CUSTOM_COMPILE_COMMAND="brev/lock-requirements.bash ${TUTORIAL}" \
        "${LOCK_IMAGE}" bash -euo pipefail -c "
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq >/dev/null
            apt-get install -y -qq --no-install-recommends ca-certificates curl gnupg >/dev/null
            curl -fsSL 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xBA6932366A755776' \
                | gpg --dearmor -o /usr/share/keyrings/deadsnakes.gpg
            echo 'deb [signed-by=/usr/share/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main' \
                > /etc/apt/sources.list.d/deadsnakes-ppa.list
            apt-get update -qq >/dev/null
            apt-get install -y -qq --no-install-recommends python${PYTHON_VERSION} python${PYTHON_VERSION}-venv >/dev/null
            python${PYTHON_VERSION} -m venv /tmp/venv
            /tmp/venv/bin/pip install -q 'pip==${PIP_VERSION}' 'pip-tools==${PIP_TOOLS_VERSION}'
            if [ -f requirements.lock.txt ]; then cp requirements.lock.txt /tmp/requirements.lock.txt; fi
            /tmp/venv/bin/pip-compile --quiet --generate-hashes --allow-unsafe --strip-extras \
                ${UPGRADE_FLAG} --output-file /tmp/requirements.lock.txt requirements.txt >&2
            cat /tmp/requirements.lock.txt
        " > "${NEW_LOCK}"

    LOCK="${REPO_ROOT}/${BREV_DIR}/requirements.lock.txt"
    if [ "${MODE}" = check ]; then
        if ! diff -u "${LOCK}" "${NEW_LOCK}" > /dev/null 2>&1; then
            echo "❌ ${BREV_DIR}/requirements.lock.txt is out of date with requirements.txt."
            echo "   Run: brev/lock-requirements.bash ${TUTORIAL}"
            diff -u "${LOCK}" "${NEW_LOCK}" | head -n 40 || true
            STATUS=1
        else
            echo "✅ ${BREV_DIR}/requirements.lock.txt is up to date."
        fi
    else
        cat "${NEW_LOCK}" > "${LOCK}"
        echo "Wrote ${BREV_DIR}/requirements.lock.txt"
    fi
    rm -f "${NEW_LOCK}"
done

exit "${STATUS}"
