#! /bin/bash
#
# Test links in markdown files and Jupyter notebooks using lychee.
#
# This script creates a temporary copy of the repo, converts notebooks to markdown,
# and runs lychee link checker to avoid polluting the local repo.
#
# Accepts one or more paths (files or directories). Notebooks are converted to
# markdown before checking; plain markdown files are checked directly.
#
# Usage:
#   ./brev/test-links.bash <path>...                          # Check paths
#   ./brev/test-links.bash tutorials/example                  # Check specific tutorial
#   ./brev/test-links.bash README.md tutorials/foo/lab.ipynb  # Check specific files
#   ./brev/test-links.bash .                                  # Check entire repo

set -eu

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") <path>...

Test links in markdown files and Jupyter notebooks using lychee.

Arguments:
  path    One or more files or directories to check

Examples:
  $(basename "$0") tutorials/example              # Check a directory
  $(basename "$0") README.md docs/guide.ipynb     # Check specific files
  $(basename "$0") .                              # Full check of entire repo

Requirements:
  - lychee must be installed and in PATH
  - jupyter nbconvert must be installed and in PATH (for notebooks)
EOF
    exit 1
}

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo -e "${RED}Error: No paths specified${NC}"
    usage
fi

# Check for required tools
if ! command -v lychee &> /dev/null; then
    echo -e "${RED}Error: lychee is not installed or not in PATH${NC}"
    echo "Please install lychee: https://github.com/lycheeverse/lychee"
    exit 1
fi

REPO_ROOT="$(pwd)"
TARGET_PATHS=("$@")

# Validate all paths exist and are inside the repo
RELATIVE_PATHS=()
HAS_NOTEBOOKS=false
for target in "${TARGET_PATHS[@]}"; do
    if [ ! -e "$target" ]; then
        echo -e "${RED}Error: Path '$target' does not exist${NC}"
        exit 1
    fi
    rel="$(realpath --relative-to="$REPO_ROOT" "$(realpath "$target")")"
    if [[ "$rel" == ../* ]]; then
        echo -e "${RED}Error: Path '$target' is not within the current repository${NC}"
        exit 1
    fi
    RELATIVE_PATHS+=("$rel")
    if [[ "$target" == *.ipynb ]] || [ -d "$target" ]; then
        HAS_NOTEBOOKS=true
    fi
done

# Only require jupyter if we may have notebooks to convert
if [ "$HAS_NOTEBOOKS" = true ]; then
    if ! command -v jupyter &> /dev/null; then
        echo -e "${RED}Error: jupyter is not installed or not in PATH${NC}"
        echo "Please install jupyter: pip install jupyter nbconvert"
        exit 1
    fi
    if ! jupyter nbconvert --version &> /dev/null; then
        echo -e "${RED}Error: jupyter nbconvert is not available${NC}"
        echo "Please install nbconvert: pip install nbconvert"
        exit 1
    fi
fi

echo "Repository: $REPO_ROOT"
echo "Checking ${#RELATIVE_PATHS[@]} path(s): ${RELATIVE_PATHS[*]}"

# Create temporary directory (notebook conversion writes files next to the
# source, so we work in a copy to avoid polluting the working tree).
TEMP_DIR=$(mktemp -d -t link-check-XXXXXX)
trap "rm -rf '$TEMP_DIR'" EXIT

TEMP_REPO="$TEMP_DIR/repo"

echo ""
echo "Creating temporary copy of repository..."
cp -r "$REPO_ROOT" "$TEMP_REPO"

# Collect notebooks that need conversion and build the lychee target list
LYCHEE_TARGETS=()
NOTEBOOKS_TO_CONVERT=()

for rel in "${RELATIVE_PATHS[@]}"; do
    temp_path="$TEMP_REPO/$rel"
    if [ -d "$temp_path" ]; then
        # Directory: find all notebooks inside it for conversion, then check
        # the whole directory.
        while IFS= read -r nb; do
            NOTEBOOKS_TO_CONVERT+=("$nb")
        done < <(find "$temp_path" -name "*.ipynb" -type f 2>/dev/null || true)
        LYCHEE_TARGETS+=("$rel")
    elif [[ "$rel" == *.ipynb ]]; then
        # Single notebook: convert it, then check the resulting .md file.
        NOTEBOOKS_TO_CONVERT+=("$temp_path")
        LYCHEE_TARGETS+=("${rel%.ipynb}.md")
    else
        # Markdown or other file: check directly.
        LYCHEE_TARGETS+=("$rel")
    fi
done

# Convert notebooks to markdown
if [ ${#NOTEBOOKS_TO_CONVERT[@]} -gt 0 ]; then
    echo ""
    echo "Converting ${#NOTEBOOKS_TO_CONVERT[@]} notebook(s) to markdown..."
    for notebook in "${NOTEBOOKS_TO_CONVERT[@]}"; do
        echo "  ${notebook#$TEMP_REPO/}"
        if ! jupyter nbconvert --to markdown "$notebook" 2>&1; then
            echo -e "${YELLOW}Warning: Failed to convert $notebook${NC}"
        fi
    done
fi

# Build lychee command
echo ""
echo "Running lychee link checker..."
echo "================================================================================"

LYCHEE_CONFIG="$REPO_ROOT/brev/lychee.toml"
LYCHEE_EXCLUDE_FILE="$REPO_ROOT/brev/.lycheeignore"

LYCHEE_CMD="lychee"
if [ -f "$LYCHEE_CONFIG" ]; then
    LYCHEE_CMD="$LYCHEE_CMD --config $LYCHEE_CONFIG"
    echo "Using lychee.toml configuration from brev/"
fi

if [ -f "$LYCHEE_EXCLUDE_FILE" ]; then
    LYCHEE_CMD="$LYCHEE_CMD --exclude-file $LYCHEE_EXCLUDE_FILE"
    echo "Using .lycheeignore file from brev/"
fi

set +e
cd "$TEMP_REPO"
$LYCHEE_CMD "${LYCHEE_TARGETS[@]}"
LYCHEE_EXIT=$?
cd "$REPO_ROOT"
set -e

echo ""
echo "================================================================================"

if [ $LYCHEE_EXIT -eq 0 ]; then
    echo -e "${GREEN}✅ All links are valid!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some links are broken!${NC}"
    exit 1
fi
