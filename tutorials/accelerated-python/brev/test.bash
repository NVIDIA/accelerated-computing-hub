#! /bin/bash
#
# Run tests for the accelerated-python tutorial.
#
# When called with no arguments, runs all package tests, the independent
# solution notebooks, and the ordered SWE application sequence.
# When called with arguments:
#   - Bare words (e.g. "06") are treated as a pytest -k filter for notebook tests.
#   - One Python test file, supplied first, is forwarded to pytest directly.
#   - Flags (e.g. "-k cupy") are applied across every test collection.
#
# Usage:
#   ./test.bash                          # run all suites
#   ./test.bash 06                       # run notebook tests matching "06"
#   ./test.bash "40 or 41"              # run notebook tests matching "40 or 41"
#   ./test.bash test/test_packages.py    # run package tests
#   ./test.bash test/test_pyhpc_packages.py # run PyHPC package tests
#   ./test.bash -k "cupy"               # filter every test collection

START_TIME=$(date +%s.%N)

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || exit 1
else
    NVIDIA_GPU_DEVICE=$(find /dev -maxdepth 1 -type c \
        -name 'nvidia[0-9]*' -print -quit 2>/dev/null)
    if [ -n "${NVIDIA_GPU_DEVICE}" ]; then
        echo "NVIDIA GPU device ${NVIDIA_GPU_DEVICE} is available; nvidia-smi is not installed"
    else
        echo "Error: no NVIDIA GPU is available" >&2
        exit 1
    fi
fi

TUTORIAL_ROOT=/accelerated-computing-hub/tutorials/accelerated-python

if [ $# -gt 0 ]; then
    FIRST_TEST_PATH=${1%%::*}
    NODE_SUFFIX=${1#"${FIRST_TEST_PATH}"}
    if [ ! -e "${FIRST_TEST_PATH}" ] && \
       [ -e "${TUTORIAL_ROOT}/${FIRST_TEST_PATH}" ]; then
        FIRST_TEST_PATH="${TUTORIAL_ROOT}/${FIRST_TEST_PATH}"
    fi
    FIRST_TEST_PATH=$(realpath -m -- "${FIRST_TEST_PATH}")

    if [ -d "${FIRST_TEST_PATH}" ]; then
        case "$(realpath -m -- "${FIRST_TEST_PATH}")" in
            "${TUTORIAL_ROOT}/test"|"${TUTORIAL_ROOT}/test/"*)
                echo "Error: directory test targets are not supported." >&2
                echo "Select one Python test file, or run this script without a path." >&2
                EXIT_CODE=2
                ;;
            *)
                echo "Error: unsupported test directory: ${FIRST_TEST_PATH}" >&2
                EXIT_CODE=2
                ;;
        esac
    elif [ -f "${FIRST_TEST_PATH}" ] && [[ "${FIRST_TEST_PATH}" == *.py ]]; then
        TEST_FILE_ALLOWED=false
        case "${FIRST_TEST_PATH}" in
            "${TUTORIAL_ROOT}/test/"*.py)
                PYTEST_ARGS=("$@")
                PYTEST_ARGS[0]="${FIRST_TEST_PATH}${NODE_SUFFIX}"
                TEST_FILE_ALLOWED=true
                ;;
            *)
                echo "Error: test file is outside ${TUTORIAL_ROOT}/test: ${FIRST_TEST_PATH}" >&2
                EXIT_CODE=2
                ;;
        esac
        if [ "${TEST_FILE_ALLOWED}" = true ]; then
            for ADDITIONAL_ARG in "${@:2}"; do
                case "${ADDITIONAL_ARG}" in
                    -*) continue ;;
                esac
                ADDITIONAL_PATH=${ADDITIONAL_ARG%%::*}
                if [ ! -e "${ADDITIONAL_PATH}" ] && \
                   [ -e "${TUTORIAL_ROOT}/${ADDITIONAL_PATH}" ]; then
                    ADDITIONAL_PATH="${TUTORIAL_ROOT}/${ADDITIONAL_PATH}"
                fi
                ADDITIONAL_PATH=$(realpath -m -- "${ADDITIONAL_PATH}")
                if [ -f "${ADDITIONAL_PATH}" ]; then
                    case "${ADDITIONAL_PATH}" in
                        "${TUTORIAL_ROOT}/test/"*.py)
                            echo "Error: select only one Python test file at a time." >&2
                            echo "Run this script without a path to test the complete suite." >&2
                            TEST_FILE_ALLOWED=false
                            EXIT_CODE=2
                            break
                            ;;
                    esac
                fi
            done
        fi
        if [ "${TEST_FILE_ALLOWED}" = true ]; then
            echo "Running: pytest ${PYTEST_ARGS[*]}"
            pytest "${PYTEST_ARGS[@]}"
            EXIT_CODE=$?
        fi
    elif [[ "$1" == -* ]]; then
        echo "Running all test collections with: $*"
        pytest \
            "${TUTORIAL_ROOT}/test/test_packages.py" \
            "${TUTORIAL_ROOT}/test/test_rapids.py" \
            "${TUTORIAL_ROOT}/test/test_pyhpc_packages.py" \
            "${TUTORIAL_ROOT}/test/test_notebooks.py" \
            "${TUTORIAL_ROOT}/test/test_swe_notebooks.py" \
            "$@"
        EXIT_CODE=$?
    elif [[ "$1" == */* ]] || [[ "$1" == *.py ]]; then
        echo "Error: test target not found: $1" >&2
        EXIT_CODE=2
    else
        echo "Running independent and ordered SWE notebook tests with -k \"$*\""
        pytest \
            "${TUTORIAL_ROOT}/test/test_notebooks.py" \
            "${TUTORIAL_ROOT}/test/test_swe_notebooks.py" \
            -k "$*"
        EXIT_CODE=$?
    fi
else
    echo "Running Accelerated Python package tests..."
    pytest \
        "${TUTORIAL_ROOT}/test/test_packages.py" \
        "${TUTORIAL_ROOT}/test/test_rapids.py" \
        "${TUTORIAL_ROOT}/test/test_pyhpc_packages.py"
    EXIT_CODE_PACKAGES=$?

    echo ""
    echo "Running independent and ordered SWE application notebook tests..."
    pytest \
        "${TUTORIAL_ROOT}/test/test_notebooks.py" \
        "${TUTORIAL_ROOT}/test/test_swe_notebooks.py"
    EXIT_CODE_NOTEBOOKS=$?

    EXIT_CODE=$((EXIT_CODE_PACKAGES || EXIT_CODE_NOTEBOOKS))
fi

END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {print $END_TIME - $START_TIME}")

echo ""
awk -v elapsed="$ELAPSED" 'BEGIN {
    hours = int(elapsed / 3600)
    minutes = int((elapsed % 3600) / 60)
    seconds = elapsed % 60
    printf "Elapsed time: %dh %dm %.3fs\n", hours, minutes, seconds
}'

exit "$EXIT_CODE"
