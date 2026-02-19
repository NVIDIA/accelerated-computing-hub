#! /bin/bash
#
# Entrypoint for the base service. Runs as root, then switches to user.

set -euo pipefail

# Generate unique TURN credentials for this container session.
# These are stored on the shared volume so both the nsight and jupyter
# services can read them.
TURN_CREDENTIALS_FILE="/accelerated-computing-hub/.turn-credentials"

TURN_USERNAME="turn_$(openssl rand -hex 8)"
TURN_PASSWORD="$(openssl rand -base64 32)"

cat > "${TURN_CREDENTIALS_FILE}" <<EOF
TURN_USERNAME=${TURN_USERNAME}
TURN_PASSWORD=${TURN_PASSWORD}
EOF

chmod 644 "${TURN_CREDENTIALS_FILE}"
echo "Generated TURN credentials in ${TURN_CREDENTIALS_FILE}"

# Switch to user and run user-level entrypoint
exec gosu "${ACH_TARGET_USER}" /accelerated-computing-hub/brev/entrypoint-base-user.bash "$@"
