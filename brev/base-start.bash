#! /bin/bash

set -euo pipefail

source /accelerated-computing-hub/brev/dev-common.bash
create_user_and_switch exec

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
  if [ -n "${ACH_TUTORIAL}" ]; then
    echo "ACH_RUN_TESTS is empty, skipping tutorial tests"
  else
    echo "ACH_TUTORIAL not set, skipping tutorial tests"
  fi
fi
