#! /bin/bash
#
# User-level entrypoint for the base service. Runs tests if configured.

set -euo pipefail

export HOME="${ACH_TARGET_HOME}"

# Generate unique TURN credentials for this container session.
# These are stored on the shared volume so the nsys, ncu, and jupyter
# services can read them.
TURN_CREDENTIALS_FILE="/accelerated-computing-hub/.turn-credentials"

if [ ! -f "${TURN_CREDENTIALS_FILE}" ]; then
  TURN_USERNAME="turn_$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 16)"
  TURN_PASSWORD="$(openssl rand -base64 48 | tr -dc 'a-zA-Z0-9' | head -c 32)"

  echo "TURN_USERNAME=${TURN_USERNAME}" > "${TURN_CREDENTIALS_FILE}"
  echo "TURN_PASSWORD=${TURN_PASSWORD}" >> "${TURN_CREDENTIALS_FILE}"

  chmod 644 "${TURN_CREDENTIALS_FILE}"
fi

# Run per-tutorial start tests if they exist.
if [ -n "${ACH_TUTORIAL:-}" ] && [ -n "${ACH_RUN_TESTS:-}" ]; then
  TEST_SCRIPT="/accelerated-computing-hub/tutorials/${ACH_TUTORIAL}/brev/test.bash"
  LOG_DIR="/accelerated-computing-hub/logs"
  LOG_FILE="${LOG_DIR}/test.log"

  # Create logs directory if it doesn't exist
  mkdir -p "${LOG_DIR}"

  if [ -f "${TEST_SCRIPT}" ]; then
    # Log with timestamp header
    {
      echo ""
      echo "=========================================="
      echo "Test Run: $(date '+%Y-%m-%d %H:%M:%S %Z')"
      echo "Tutorial: ${ACH_TUTORIAL}"
      echo "=========================================="
    } | tee -a "${LOG_FILE}"

    # Run tests with output to both console and log file
    if bash "${TEST_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"; then
      {
        echo "=========================================="
        echo "Tests completed successfully for: ${ACH_TUTORIAL}"
        echo "=========================================="
        echo ""
      } | tee -a "${LOG_FILE}"
    else
      {
        echo "=========================================="
        echo "Tests FAILED for: ${ACH_TUTORIAL}"
        echo "=========================================="
        echo ""
      } | tee -a "${LOG_FILE}"
      exit 1
    fi
  else
    {
      echo "No test script found at: ${TEST_SCRIPT}"
      echo "Skipping tests for tutorial: ${ACH_TUTORIAL}"
    } | tee -a "${LOG_FILE}"
  fi
else
  if [ -n "${ACH_TUTORIAL:-}" ]; then
    echo "ACH_RUN_TESTS is empty, skipping tutorial tests"
  else
    echo "ACH_TUTORIAL not set, skipping tutorial tests"
  fi
fi
