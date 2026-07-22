#! /bin/bash
#
# Entrypoint for interactive shell sessions.

set -euo pipefail

if [ "$(id -u)" = "0" ] && [ "${ACH_TARGET_USER}" != "$(id -un)" ]; then
    exec gosu "${ACH_TARGET_USER}" bash -l
else
    exec bash -l
fi
