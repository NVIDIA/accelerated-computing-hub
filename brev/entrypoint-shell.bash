#! /bin/bash
#
# Entrypoint for interactive shell sessions.

set -euo pipefail

if [ "$(id -u)" = "0" ]; then
    exec gosu "${ACH_TARGET_USER}" bash -l
fi

exec bash -l
