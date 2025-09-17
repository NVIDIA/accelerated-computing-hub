#!/bin/bash

JUPYTER_HOST="jupyter0-${BREV_ENV_ID}.brevlab.com"
NSIGHT_HTTP_URL="https://nsight0-${BREV_ENV_ID}.brevlab.com"

cat << EOF
{
  "ui": {
    "enabled": true,
    "suppressServerAddressWarning": true,
    "host": "${JUPYTER_HOST}",
    "dockerHost": "${JUPYTER_HOST}",
    "defaultStreamerAddress": "${NSIGHT_HTTP_URL}"
  }
}
EOF
