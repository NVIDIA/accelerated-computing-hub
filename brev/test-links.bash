#! /bin/bash
#
# Test links in markdown files and Jupyter notebooks using lychee.
#
# This script creates a temporary copy of the repo, converts notebooks to markdown,
# and runs lychee link checker to avoid polluting the local repo.
#
# Usage:
#   ./brev/test-links.bash <path>              # Check specific path
#   ./brev/test-links.bash tutorials/example   # Check specific tutorial
#   ./brev/test-links.bash .                   # Check entire repo

set -eu

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") <path>

Test links in markdown files and Jupyter notebooks using lychee.

Arguments:
  path    Path to check (file, directory, or tutorial)

Examples:
  $(basename "$0") tutorials/example    # Check specific tutorial
  $(basename "$0") .                    # Check entire repo

Requirements:
  - lychee must be installed and in PATH
  - jupyter nbconvert must be installed and in PATH
EOF
    exit 1
}

# Check if path argument is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No path specified${NC}"
    usage
fi

TARGET_PATH="$1"

# Get absolute paths
REPO_ROOT="$(pwd)"
TARGET_ABS="$(cd "$TARGET_PATH" && pwd)" 2>/dev/null || {
    echo -e "${RED}Error: Path '$TARGET_PATH' does not exist${NC}"
    exit 1
}

# Get relative path from repo root
RELATIVE_TARGET="$(realpath --relative-to="$REPO_ROOT" "$TARGET_ABS")"

# Check if path is outside repo (would start with ../)
if [[ "$RELATIVE_TARGET" == ../* ]]; then
    echo -e "${RED}Error: Path '$TARGET_PATH' is not within the current repository${NC}"
    exit 1
fi

echo "Repository: $REPO_ROOT"
echo "Target path: $RELATIVE_TARGET"

# Create temporary directory
TEMP_DIR=$(mktemp -d -t link-check-XXXXXX)
trap "rm -rf '$TEMP_DIR'" EXIT

TEMP_REPO="$TEMP_DIR/repo"

echo ""
echo "Creating temporary copy of repository in: $TEMP_REPO"
echo "Copying files..."

# Copy the repo to temp directory
cp -r "$REPO_ROOT" "$TEMP_REPO"

# Find and convert notebooks in the temp copy
TEMP_TARGET="$TEMP_REPO/$RELATIVE_TARGET"

echo ""
echo "Looking for Jupyter notebooks to convert..."
NOTEBOOKS=$(find "$TEMP_TARGET" -name "*.ipynb" -type f 2>/dev/null || true)

if [ -z "$NOTEBOOKS" ]; then
    echo "No notebooks found in $RELATIVE_TARGET"
else
    NOTEBOOK_COUNT=$(echo "$NOTEBOOKS" | wc -l)
    echo "Found $NOTEBOOK_COUNT notebook(s) to convert:"
    echo "$NOTEBOOKS" | sed 's/^/  - /'

    echo ""
    echo "Converting notebooks to markdown..."

    while IFS= read -r notebook; do
        echo "Converting: ${notebook#$TEMP_REPO/}"
        if ! jupyter nbconvert --to markdown "$notebook" 2>&1; then
            echo -e "${YELLOW}Warning: Failed to convert $notebook${NC}"
        fi
    done <<< "$NOTEBOOKS"
fi

# Run lychee directly
echo ""
echo "Running lychee link checker on: $RELATIVE_TARGET"
echo "================================================================================"

# Reference lychee config files from brev directory
LYCHEE_CONFIG="$REPO_ROOT/brev/lychee.toml"
LYCHEE_EXCLUDE_FILE="$REPO_ROOT/brev/.lycheeignore"

# Build lychee command with config options
LYCHEE_CMD="lychee"
if [ -f "$LYCHEE_CONFIG" ]; then
    LYCHEE_CMD="$LYCHEE_CMD --config $LYCHEE_CONFIG"
    echo "Using lychee.toml configuration from brev/"
fi

if [ -f "$LYCHEE_EXCLUDE_FILE" ]; then
    LYCHEE_CMD="$LYCHEE_CMD --exclude-file $LYCHEE_EXCLUDE_FILE"
    echo "Using .lycheeignore file from brev/"
fi

# Change to temp repo directory and run lychee on the target path
# Lychee will recursively find all markdown files
set +e
cd "$TEMP_REPO"
$LYCHEE_CMD "$RELATIVE_TARGET"
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
