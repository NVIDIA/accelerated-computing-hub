#! /bin/bash
#
# Generate a CSCS Slurm Container Engine EDF for a tutorial image.
#
# The EDF can be used on CSCS systems with Slurm's container support, e.g.:
#   srun --environment=${SCRATCH}/ach/accelerated-python.toml bash

set -eu

SCRIPT_PATH=$(cd "$(dirname "${0}")"; pwd -P)
REPO_ROOT=$(cd "${SCRIPT_PATH}/.."; pwd -P)

usage() {
    cat << USAGE
Usage: $(basename "$0") [options] <tutorial-name|tutorial-dir|docker-compose-file>

Options:
  --image IMAGE       Container image URI (default: image from docker-compose.yml)
  --tag TAG           Replace the image tag before writing the EDF
  --output FILE       EDF output path (default: <tutorial>/brev/cscs.toml)
  --mount             Bind-mount this repository at /accelerated-computing-hub
  --no-mount          Do not bind-mount this repository (default)
  --workdir DIR       Working directory inside the container (default: compose working_dir)

Generate a CSCS Slurm Container Engine EDF for use on systems where the
interactive Docker Compose stack is not available.
USAGE
    exit 1
}

IMAGE=""
IMAGE_TAG=""
OUTPUT=""
MOUNT=false
WORKDIR=""

format_cscs_image_ref() {
    local image_ref=$1
    local registry=${image_ref%%/*}

    if [[ "${image_ref}" == /* ]] || [[ "${image_ref}" == *#* ]]; then
        echo "${image_ref}"
    elif [[ "${image_ref}" == */* ]] && \
         { [[ "${registry}" == *.* ]] || [[ "${registry}" == *:* ]] || \
           [ "${registry}" = "localhost" ]; }; then
        echo "${registry}#${image_ref#*/}"
    else
        echo "${image_ref}"
    fi
}

replace_image_tag() {
    local image_ref=$1
    local replacement_tag=$2
    local digest=""
    local name="${image_ref}"
    local final_segment

    if [[ "${name}" == *@* ]]; then
        digest="@${name#*@}"
        name="${name%@*}"
    fi
    final_segment="${name##*/}"
    final_segment="${final_segment##*#}"
    if [[ "${final_segment}" == *:* ]]; then
        name="${name%:*}"
    fi
    echo "${name}:${replacement_tag}${digest}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --tag) IMAGE_TAG="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --mount) MOUNT=true; shift ;;
        --no-mount) MOUNT=false; shift ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        --*) echo "Error: unknown option $1" >&2; usage ;;
        *) break ;;
    esac
done

[ $# -eq 1 ] || usage
ARG=$1

if [[ "${ARG}" == *"/"* ]]; then
    ARG_PATH="${ARG}"
    [[ "${ARG_PATH}" == /* ]] || ARG_PATH="${REPO_ROOT}/${ARG_PATH}"
    if [ -d "${ARG_PATH}" ]; then
        TUTORIAL_DIR="${ARG_PATH}"
        COMPOSE_FILE="${TUTORIAL_DIR}/brev/docker-compose.yml"
    else
        COMPOSE_FILE="${ARG_PATH}"
        TUTORIAL_DIR=$(dirname "$(dirname "${COMPOSE_FILE}")")
    fi
    ACH_TUTORIAL=$(basename "${TUTORIAL_DIR}")
else
    ACH_TUTORIAL="${ARG}"
    TUTORIAL_DIR="${REPO_ROOT}/tutorials/${ACH_TUTORIAL}"
    COMPOSE_FILE="${TUTORIAL_DIR}/brev/docker-compose.yml"
fi

if [ ! -f "${COMPOSE_FILE}" ]; then
    echo "Error: compose file not found: ${COMPOSE_FILE}" >&2
    exit 1
fi

# Generated syllabus Compose files are nested below a tutorial's brev/
# directory, so dirname alone cannot identify the tutorial. Read the
# repository's tutorial-name anchor even when --image and --workdir let this
# command run without Docker Compose.
COMPOSE_TUTORIAL=$(sed -n \
    's/^name: &tutorial-name \([^[:space:]#]*\).*$/\1/p' \
    "${COMPOSE_FILE}" | head -n1)
if [ -n "${COMPOSE_TUTORIAL}" ]; then
    ACH_TUTORIAL=${COMPOSE_TUTORIAL}
    COMPOSE_PARENT=$(dirname "${COMPOSE_FILE}")
    while [ "${COMPOSE_PARENT}" != "/" ] && \
          [ "$(basename "${COMPOSE_PARENT}")" != "${ACH_TUTORIAL}" ]; do
        COMPOSE_PARENT=$(dirname "${COMPOSE_PARENT}")
    done
    if [ "$(basename "${COMPOSE_PARENT}")" = "${ACH_TUTORIAL}" ]; then
        TUTORIAL_DIR=${COMPOSE_PARENT}
    fi
fi

if [ -z "${IMAGE}" ] || [ -z "${WORKDIR}" ]; then
    if ! command -v docker >/dev/null 2>&1 || \
       ! docker compose version >/dev/null 2>&1; then
        echo "Error: Docker Compose is required to infer the image or working directory." >&2
        echo "Pass both --image and --workdir when generating an EDF without Docker Compose." >&2
        exit 1
    fi
    if ! command -v jq >/dev/null 2>&1; then
        echo "Error: jq is required to read ${COMPOSE_FILE}." >&2
        echo "Install jq, or pass both --image and --workdir." >&2
        exit 1
    fi
    COMPOSE_CONFIG=$(docker compose -f "${COMPOSE_FILE}" config --format json)
fi
if [ -n "${COMPOSE_CONFIG:-}" ]; then
    COMPOSE_TUTORIAL=$(jq -r '
        ."x-config"."common-env".ACH_TUTORIAL //
        .services.base.environment.ACH_TUTORIAL // empty
    ' <<< "${COMPOSE_CONFIG}")
    if [ -n "${COMPOSE_TUTORIAL}" ]; then
        ACH_TUTORIAL=${COMPOSE_TUTORIAL}
    fi
fi
if [ -z "${IMAGE}" ]; then
    IMAGE=$(jq -r '."x-config".image // empty' <<< "${COMPOSE_CONFIG}")
    if [ -z "${IMAGE}" ]; then
        echo "Error: x-config.image is not set in ${COMPOSE_FILE}" >&2
        exit 1
    fi
fi
if [ -n "${IMAGE_TAG}" ]; then
    IMAGE=$(replace_image_tag "${IMAGE}" "${IMAGE_TAG}")
fi
IMAGE=$(format_cscs_image_ref "${IMAGE}")
if [ -z "${WORKDIR}" ]; then
    WORKDIR=$(jq -r '."x-config"."working-dir" // empty' <<< "${COMPOSE_CONFIG}")
fi
if [ -z "${OUTPUT}" ]; then
    OUTPUT="${TUTORIAL_DIR}/brev/cscs.toml"
fi

mkdir -p "$(dirname "${OUTPUT}")"
{
    echo "# Generated by accelerated-computing-hub/brev/generate-cscs-edf.bash"
    echo "# Tutorial: ${ACH_TUTORIAL}"
    echo "image = \"${IMAGE}\""
    echo "workdir = \"${WORKDIR:-/accelerated-computing-hub}\""
    echo "mounts = ["
    if [ "${MOUNT}" = "true" ]; then
        echo "  \"${REPO_ROOT}:/accelerated-computing-hub\","
    fi
    echo "]"
    echo "[env]"
    echo "ACH_TUTORIAL = \"${ACH_TUTORIAL}\""
    echo "ACH_USER = \"${ACH_USER:-ach}\""
    echo "ACH_UID = \"${ACH_UID:-1000}\""
    echo "ACH_GID = \"${ACH_GID:-1000}\""
    if [ "${ACH_TUTORIAL}" = "accelerated-python" ]; then
        echo "PMIX_MCA_gds = \"hash\""
        echo "PMIX_MCA_psec = \"native\""
    fi
} > "${OUTPUT}"

echo "Generated CSCS EDF: ${OUTPUT}"
echo "Image: ${IMAGE}"
