#! /bin/bash

set -eu

# Run per-tutorial start tests if they exist.
if [ -n "${ACH_TUTORIAL}" ]; then
  TEST_SCRIPT="/accelerated-computing-hub/tutorials/${ACH_TUTORIAL}/brev/test-start.bash"
  LOG_DIR="/accelerated-computing-hub/logs"
  LOG_FILE="${LOG_DIR}/test-start.log"

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
  echo "ACH_TUTORIAL not set, skipping tutorial tests"
fi
