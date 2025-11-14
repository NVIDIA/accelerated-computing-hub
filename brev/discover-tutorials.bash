#! /bin/bash

# This script discovers all tutorial directories in the repository.
# It outputs one tutorial directory path per line (relative to repo root).

set -eu

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
REPO_ROOT=$(cd ${SCRIPT_PATH}/..; pwd -P)

# Find all directories under tutorials/ at depth 1 that have brev/docker-compose.yml
cd "${REPO_ROOT}"
for dir in tutorials/*/; do
    if [ -f "${dir}brev/docker-compose.yml" ]; then
        echo "${dir%/}"
    fi
done | sort
