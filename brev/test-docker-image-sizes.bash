#!/usr/bin/env bash
#
# Query and compare Docker image sizes from GHCR.
#
# Usage:
#   test-docker-image-sizes.bash compare --tutorial <name|all> --tag <tag> [--baseline <tag>] [--format text|markdown|json]
#   test-docker-image-sizes.bash history --tutorial <name|all> [--count <N>] [--filter <prefix>] [--format text|markdown|json]
#
# Commands:
#   compare   Compare the compressed size of an image tag against a baseline
#   history   Show compressed sizes of the last N image tags
#
# Options:
#   --owner <owner>       GHCR image owner (default: $GITHUB_REPOSITORY_OWNER or 'nvidia')
#   --tutorial <name>     Tutorial name, or 'all' to discover and test every tutorial
#   --tag <tag>           Image tag to check (compare mode)
#   --baseline <tag>      Baseline tag to compare against (default: 'main-latest')
#   --count <N>           Number of recent tags to show (default: 10)
#   --filter <prefix>     Tag prefix filter for history (default: 'main-git-')
#   --format <fmt>        Output format: text, markdown, json (default: text)
#
# Environment:
#   GITHUB_TOKEN              Token for GHCR authentication (uses gh auth token as fallback)
#   GITHUB_REPOSITORY_OWNER   Default image owner
#
# Examples:
#   # Compare all tutorials against main
#   test-docker-image-sizes.bash compare --tutorial all \
#     --tag pull-request-42-latest --baseline main-latest
#
#   # Compare a single tutorial
#   test-docker-image-sizes.bash compare --tutorial accelerated-python \
#     --tag pull-request-42-git-abc1234 --baseline main-latest
#
#   # Show last 5 main builds for all tutorials
#   test-docker-image-sizes.bash history --tutorial all --count 5

set -euo pipefail

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

REGISTRY="ghcr.io"
OWNER="${GITHUB_REPOSITORY_OWNER:-nvidia}"
OWNER="${OWNER,,}"
BASELINE_TAG="main-latest"
HISTORY_COUNT=10
HISTORY_FILTER="main-git-"
FORMAT="text"
TUTORIAL=""
TAG=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    sed -n '2,/^$/{ s/^# \{0,1\}//; p; }' "$0"
    exit "${1:-0}"
}

# ---------- GHCR helpers ----------

# Obtain a short-lived pull token for the GHCR OCI Distribution API.
ghcr_token() {
    local repo=$1
    local auth_flag=""

    if [ -n "${GITHUB_TOKEN:-}" ]; then
        auth_flag="-u _:${GITHUB_TOKEN}"
    elif command -v gh &>/dev/null; then
        local gh_token
        gh_token=$(gh auth token 2>/dev/null || true)
        if [ -n "${gh_token}" ]; then
            auth_flag="-u _:${gh_token}"
        fi
    fi

    local response
    # auth_flag intentionally unquoted so it's omitted when empty
    response=$(curl -s --connect-timeout 10 --max-time 30 ${auth_flag} \
        "https://${REGISTRY}/token?service=${REGISTRY}&scope=repository:${repo}:pull") || {
        echo "Error: Could not reach ${REGISTRY} token endpoint" >&2
        return 1
    }

    local token
    token=$(echo "$response" | jq -r '.token // empty')
    if [ -z "$token" ]; then
        echo "Error: Failed to get GHCR token for ${repo}" >&2
        echo "  Response: ${response}" >&2
        return 1
    fi
    echo "$token"
}

# Fetch a manifest (or manifest index) from GHCR.
ghcr_manifest() {
    local token=$1 repo=$2 ref=$3

    curl -sf --connect-timeout 10 --max-time 30 \
        -H "Authorization: Bearer ${token}" \
        -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json" \
        "https://${REGISTRY}/v2/${repo}/manifests/${ref}"
}

# Return the compressed image size (bytes) for a single tag.
# Resolves manifest indexes to the linux/amd64 image manifest.
get_image_size() {
    local token=$1 repo=$2 tag=$3

    local manifest
    manifest=$(ghcr_manifest "$token" "$repo" "$tag") || {
        echo "Error: Could not fetch manifest for ${repo}:${tag}" >&2
        return 1
    }

    local media_type
    media_type=$(echo "$manifest" | jq -r '.mediaType // empty')

    if [[ "$media_type" == *"index"* ]] || [[ "$media_type" == *"manifest.list"* ]]; then
        local digest
        digest=$(echo "$manifest" | jq -r '
            [.manifests[] |
             select(.platform.architecture == "amd64" and .platform.os == "linux")] |
            first | .digest // empty')

        if [ -z "$digest" ]; then
            echo "Error: No linux/amd64 manifest in index for ${repo}:${tag}" >&2
            return 1
        fi

        manifest=$(ghcr_manifest "$token" "$repo" "$digest") || {
            echo "Error: Could not resolve inner manifest for ${repo}:${tag}" >&2
            return 1
        }
    fi

    local size
    size=$(echo "$manifest" | jq '[.config.size, (.layers[].size)] | add')

    if [ -z "$size" ] || [ "$size" = "null" ]; then
        echo "Error: Could not compute size from manifest for ${repo}:${tag}" >&2
        return 1
    fi
    echo "$size"
}

# ---------- Formatting ----------

format_bytes() {
    local bytes=$1
    awk -v b="$bytes" 'BEGIN {
        ab = (b < 0) ? -b : b
        prefix = (b < 0) ? "-" : ""
        if      (ab >= 1073741824) printf "%s%.2f GB", prefix, ab / 1073741824
        else if (ab >= 1048576)    printf "%s%.2f MB", prefix, ab / 1048576
        else if (ab >= 1024)       printf "%s%.2f KB", prefix, ab / 1024
        else                       printf "%s%d B",    prefix, ab
    }'
}

# ---------- Tutorial discovery ----------

discover_tutorials() {
    "${SCRIPT_PATH}/discover-tutorials.bash" | while read -r path; do
        basename "$path"
    done
}

# ---------- Commands ----------

# Compare a single tutorial. Outputs one result object/block.
# Returns non-zero if the current tag could not be fetched.
compare_one() {
    local tutorial=$1

    local image="${tutorial}-tutorial"
    local repo="${OWNER}/${image}"

    local token
    token=$(ghcr_token "$repo")

    local current_size
    current_size=$(get_image_size "$token" "$repo" "$TAG") || {
        echo "Error: Could not get size for ${repo}:${TAG}" >&2
        return 1
    }

    local baseline_exists=true baseline_size
    baseline_size=$(get_image_size "$token" "$repo" "$BASELINE_TAG" 2>/dev/null) || baseline_exists=false

    local diff_bytes=0 diff_pct="0.00" sign=""
    local change_human=""
    if $baseline_exists; then
        diff_bytes=$((current_size - baseline_size))
        if [ "$baseline_size" -ne 0 ]; then
            diff_pct=$(awk "BEGIN { printf \"%.2f\", (${diff_bytes} / ${baseline_size}) * 100 }")
        fi
        if [ "$diff_bytes" -gt 0 ]; then sign="+"; fi
        change_human="${sign}$(format_bytes "$diff_bytes") (${sign}${diff_pct}%)"
    fi

    case "$FORMAT" in
    json)
        local be_json="true" bs_json="$baseline_size" bsh_json db_json dp_json
        if ! $baseline_exists; then
            be_json="false"; bs_json="null"; bsh_json=""; db_json="null"; dp_json="null"
        else
            bsh_json=$(format_bytes "$baseline_size"); db_json="$diff_bytes"; dp_json="$diff_pct"
        fi

        jq -n \
            --arg tutorial "$tutorial" \
            --arg image "${REGISTRY}/${repo}" \
            --arg current_tag "$TAG" \
            --argjson current_size "$current_size" \
            --arg current_size_human "$(format_bytes "$current_size")" \
            --arg baseline_tag "$BASELINE_TAG" \
            --argjson baseline_exists "$be_json" \
            --argjson baseline_size "$bs_json" \
            --arg baseline_size_human "$bsh_json" \
            --argjson diff_bytes "$db_json" \
            --argjson diff_pct "$dp_json" \
            --arg change_human "$change_human" \
            '{
                tutorial: $tutorial,
                image: $image,
                current_tag: $current_tag,
                current_size: $current_size,
                current_size_human: $current_size_human,
                baseline_tag: $baseline_tag,
                baseline_exists: $baseline_exists,
                baseline_size: $baseline_size,
                baseline_size_human: $baseline_size_human,
                diff_bytes: $diff_bytes,
                diff_pct: $diff_pct,
                change_human: $change_human
            }'
        ;;
    markdown)
        echo "### ${tutorial}"
        echo ""
        echo "| | Tag | Compressed Size |"
        echo "|---|---|---:|"
        echo "| Current | \`${TAG}\` | $(format_bytes "$current_size") |"
        if $baseline_exists; then
            echo "| Baseline | \`${BASELINE_TAG}\` | $(format_bytes "$baseline_size") |"
            echo "| **Change** | | **${change_human}** |"
        else
            echo "| Baseline | \`${BASELINE_TAG}\` | _(not found)_ |"
        fi
        echo ""
        ;;
    *)
        echo "========================================"
        echo "Image: ${REGISTRY}/${repo}"
        echo "========================================"
        printf "  %-10s %-40s %s\n" "Current" "(${TAG})" "$(format_bytes "$current_size")"
        if $baseline_exists; then
            printf "  %-10s %-40s %s\n" "Baseline" "(${BASELINE_TAG})" "$(format_bytes "$baseline_size")"
            echo "  ────────────────────────────────────────────────────"
            printf "  Change:  %s\n" "$change_human"
        else
            printf "  %-10s %-40s %s\n" "Baseline" "(${BASELINE_TAG})" "(not found)"
        fi
        echo ""
        ;;
    esac
}

cmd_compare() {
    if [ -z "$TUTORIAL" ]; then echo "Error: --tutorial is required" >&2; exit 1; fi
    if [ -z "$TAG" ];      then echo "Error: --tag is required for compare" >&2; exit 1; fi

    if [ "$TUTORIAL" != "all" ]; then
        compare_one "$TUTORIAL"
        return
    fi

    # --tutorial all: discover and compare every tutorial
    local tutorials
    tutorials=$(discover_tutorials)

    local json_results="[]"

    for tutorial in ${tutorials}; do
        local result
        if result=$(compare_one "$tutorial" 2>/dev/null); then
            case "$FORMAT" in
            json) json_results=$(echo "$json_results" | jq --argjson r "$result" '. + [$r]') ;;
            *)    echo "$result" ;;
            esac
        else
            case "$FORMAT" in
            json) json_results=$(echo "$json_results" | jq --arg t "$tutorial" '. + [{"tutorial": $t, "error": true}]') ;;
            markdown)
                echo "### ${tutorial}"
                echo ""
                echo "> ⚠️ Could not fetch image size (image may not exist yet)"
                echo ""
                ;;
            *)
                echo "========================================"
                echo "  ${tutorial}: could not fetch image size"
                echo "========================================"
                echo ""
                ;;
            esac
        fi
    done

    if [ "$FORMAT" = "json" ]; then
        echo "$json_results"
    fi
}

history_one() {
    local tutorial=$1

    if ! command -v gh &>/dev/null; then
        echo "Error: The 'gh' CLI is required for history mode" >&2
        return 1
    fi

    local image="${tutorial}-tutorial"
    local repo="${OWNER}/${image}"

    local token
    token=$(ghcr_token "$repo")

    local versions=""
    local jq_filter
    jq_filter=$(cat <<'JQ'
        [flatten[]
         | select(.metadata.container.tags | any(startswith($filter)))
         | {tags: .metadata.container.tags, created: .created_at}]
        | sort_by(.created) | reverse
        | .[0:($count | tonumber)]
JQ
    )

    versions=$(gh api "/orgs/${OWNER}/packages/container/${image}/versions" \
        --paginate 2>/dev/null \
        | jq -s --arg filter "$HISTORY_FILTER" --arg count "$HISTORY_COUNT" "$jq_filter") \
    || versions=$(gh api "/users/${OWNER}/packages/container/${image}/versions" \
        --paginate 2>/dev/null \
        | jq -s --arg filter "$HISTORY_FILTER" --arg count "$HISTORY_COUNT" "$jq_filter") \
    || {
        echo "Error: Could not list package versions for ${image}" >&2
        return 1
    }

    local count
    count=$(echo "$versions" | jq 'length')

    if [ "$count" -eq 0 ]; then
        echo "No tags matching '${HISTORY_FILTER}*' found for ${image}"
        return
    fi

    local json_out="[]"

    case "$FORMAT" in
    markdown)
        echo "### ${tutorial}"
        echo ""
        echo "| Tag | Compressed Size | Date |"
        echo "|---|---:|---|"
        ;;
    json) ;;
    *)
        echo "========================================"
        echo "Image: ${REGISTRY}/${repo}"
        echo "Filter: ${HISTORY_FILTER}*"
        echo "========================================"
        printf "  %-40s  %-12s  %s\n" "Tag" "Size" "Date"
        echo "  ──────────────────────────────────────────────────────────────────"
        ;;
    esac

    for i in $(seq 0 $((count - 1))); do
        local entry tag created size_str
        entry=$(echo "$versions" | jq ".[$i]")
        tag=$(echo "$entry" | jq -r \
            "[.tags[] | select(startswith(\"${HISTORY_FILTER}\"))] | first")
        created=$(echo "$entry" | jq -r '.created' | cut -dT -f1)

        local size
        if size=$(get_image_size "$token" "$repo" "$tag" 2>/dev/null); then
            size_str=$(format_bytes "$size")
        else
            size_str="N/A"
            size="null"
        fi

        case "$FORMAT" in
        json)
            json_out=$(echo "$json_out" | jq \
                --arg tag "$tag" \
                --argjson size "$size" \
                --arg size_human "$size_str" \
                --arg date "$created" \
                '. + [{tag: $tag, size: $size, size_human: $size_human, date: $date}]')
            ;;
        markdown)
            echo "| \`${tag}\` | ${size_str} | ${created} |"
            ;;
        *)
            printf "  %-40s  %-12s  %s\n" "$tag" "$size_str" "$created"
            ;;
        esac
    done

    case "$FORMAT" in
    json)     echo "$json_out" | jq --arg tutorial "$tutorial" \
                  --arg image "${REGISTRY}/${repo}" \
                  '{tutorial: $tutorial, image: $image, entries: .}' ;;
    markdown) echo "" ;;
    esac
    echo ""
}

cmd_history() {
    if [ -z "$TUTORIAL" ]; then echo "Error: --tutorial is required" >&2; exit 1; fi

    if [ "$TUTORIAL" != "all" ]; then
        history_one "$TUTORIAL"
        return
    fi

    local tutorials
    tutorials=$(discover_tutorials)

    local json_results="[]"

    for tutorial in ${tutorials}; do
        local result
        if result=$(history_one "$tutorial" 2>/dev/null); then
            case "$FORMAT" in
            json) json_results=$(echo "$json_results" | jq --argjson r "$result" '. + [$r]') ;;
            *)    echo "$result" ;;
            esac
        else
            case "$FORMAT" in
            json) json_results=$(echo "$json_results" | jq --arg t "$tutorial" '. + [{"tutorial": $t, "error": true}]') ;;
            *)    echo "${tutorial}: could not fetch history" >&2 ;;
            esac
        fi
    done

    if [ "$FORMAT" = "json" ]; then
        echo "$json_results"
    fi
}

# ---------- Argument parsing ----------

COMMAND="${1:-}"
shift 2>/dev/null || true

while [ $# -gt 0 ]; do
    case "$1" in
        --owner)    OWNER="${2,,}"; shift 2 ;;
        --tutorial) TUTORIAL="$2";  shift 2 ;;
        --tag)      TAG="$2";       shift 2 ;;
        --baseline) BASELINE_TAG="$2"; shift 2 ;;
        --count)    HISTORY_COUNT="$2"; shift 2 ;;
        --filter)   HISTORY_FILTER="$2"; shift 2 ;;
        --format)   FORMAT="$2";    shift 2 ;;
        -h|--help)  usage 0 ;;
        *)          echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

case "$COMMAND" in
    compare)       cmd_compare ;;
    history)       cmd_history ;;
    -h|--help|"")  usage 0 ;;
    *)             echo "Unknown command: ${COMMAND}" >&2; usage 1 ;;
esac
