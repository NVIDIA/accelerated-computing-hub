#! /bin/bash
#
# Keep each tutorial's pip requirements file in sync with its uv.lock.
#
# Every exported file starts with the `uv export` command that produced it. This
# script re-runs that command from the file's directory, so the header is the
# single place that says how a file is generated.
#
# Usage:
#   brev/update-lock-exports.bash          Regenerate the exported files.
#   brev/update-lock-exports.bash --check  Fail if a uv.lock is out of date with
#                                          its pyproject.toml, or an exported
#                                          file is out of date with its uv.lock.

set -eu -o pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

CHECK=0
if [ "${1:-}" = "--check" ]; then
    CHECK=1
elif [ $# -gt 0 ]; then
    echo "Usage: $0 [--check]" >&2
    exit 2
fi

STATUS=0

if [ "${CHECK}" -eq 1 ]; then
    while IFS= read -r lock; do
        project=$(dirname "${lock}")
        if ! uv lock --check --quiet --project "${project}"; then
            echo "✗ ${lock} is out of date with ${project}/pyproject.toml; run: (cd ${project} && uv lock)" >&2
            STATUS=1
        fi
    done < <(git ls-files -- '*/uv.lock')
fi

# Find exported files by the header uv writes on their second line.
while IFS= read -r file; do
    header=$(sed -n '2p' "${file}")
    [[ "${header}" =~ ^\#\ +(uv\ export\ .*)$ ]] || continue
    command="${BASH_REMATCH[1]}"
    directory=$(dirname "${file}")
    output=$(basename "${file}")

    if [ "${CHECK}" -eq 1 ]; then
        expected=$(mktemp)
        # Write next to the original so the recorded `-o` path resolves the same way.
        (cd "${directory}" && eval "${command/-o ${output}/-o ${expected}} --quiet")
        # The header names the temporary file; compare everything after it.
        if ! diff -q <(tail -n +3 "${file}") <(tail -n +3 "${expected}") >/dev/null; then
            echo "✗ ${file} is out of date with its uv.lock; run: brev/update-lock-exports.bash" >&2
            STATUS=1
        else
            echo "✓ ${file}"
        fi
        rm -f "${expected}"
    else
        (cd "${directory}" && eval "${command} --quiet")
        echo "Updated ${file}"
    fi
done < <(git ls-files -- '*requirements*.txt')

exit "${STATUS}"
