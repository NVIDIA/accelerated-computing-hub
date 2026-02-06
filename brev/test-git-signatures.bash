#! /bin/bash
#
# Test that commits are properly signed.
#
# This script checks that commits on the current branch (compared to origin/main)
# have valid signatures. It uses git verify-commit for local verification.
#
# Note: In CI, the GitHub API is used instead for more comprehensive verification
# (including web-based commits and SSH signatures).
#
# Usage:
#   ./brev/test-git-signatures.bash              # Check commits since origin/main
#   ./brev/test-git-signatures.bash <base-ref>   # Check commits since specified ref

set -eu

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") [base-ref]

Test that commits are properly signed.

Arguments:
  base-ref    Base reference to compare against (optional, defaults to origin/main)

Examples:
  $(basename "$0")                    # Check commits since origin/main
  $(basename "$0") origin/develop     # Check commits since origin/develop
  $(basename "$0") HEAD~5             # Check last 5 commits

This script checks that all commits between the base ref and HEAD are signed.

Requirements:
  - Git must be installed
  - For local verification, GPG must be configured for signature verification
EOF
    exit 1
}

# Parse arguments
BASE_REF="origin/main"
if [ $# -gt 0 ]; then
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        usage
    fi
    BASE_REF="$1"
fi

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
echo "Current branch: $CURRENT_BRANCH"

# Skip signature check on main branch
if [ "$CURRENT_BRANCH" = "main" ]; then
    echo -e "${YELLOW}On main branch - skipping signature check for historical commits.${NC}"
    echo "Signature verification only runs on PR branches."
    exit 0
fi

# Get commits that are on this branch but not on base ref
echo "Comparing against: $BASE_REF"
echo ""

COMMIT_SHAS=""
if git rev-parse "$BASE_REF" >/dev/null 2>&1; then
    COMMIT_SHAS=$(git rev-list "$BASE_REF..HEAD" 2>/dev/null || true)
else
    echo -e "${YELLOW}Warning: Could not find $BASE_REF, checking only HEAD commit.${NC}"
    COMMIT_SHAS=$(git rev-parse HEAD)
fi

if [ -z "$COMMIT_SHAS" ]; then
    echo "No new commits to check (branch is up to date with $BASE_REF)."
    exit 0
fi

COMMIT_COUNT=$(echo "$COMMIT_SHAS" | wc -l)
echo "Checking $COMMIT_COUNT commit(s) for signatures..."
echo ""

# Check each commit
UNSIGNED_COMMITS=""
UNSIGNED_COUNT=0

for sha in $COMMIT_SHAS; do
    SHORT_SHA="${sha:0:7}"
    MESSAGE=$(git log -1 --format="%s" "$sha" 2>/dev/null || echo "unknown")

    # Try to verify the commit signature
    if git verify-commit "$sha" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $SHORT_SHA - $MESSAGE (signed)"
    else
        # Check if it has a signature at all (even if we can't verify it)
        if git log -1 --format="%G?" "$sha" 2>/dev/null | grep -q "^[GUX]"; then
            # G = good, U = good but untrusted, X = good but expired
            SIGNATURE_STATUS=$(git log -1 --format="%G?" "$sha" 2>/dev/null)
            echo -e "${YELLOW}⚠${NC} $SHORT_SHA - $MESSAGE (signature present but unverifiable: $SIGNATURE_STATUS)"
        else
            echo -e "${RED}✗${NC} $SHORT_SHA - $MESSAGE (NOT signed)"
            UNSIGNED_COMMITS="$UNSIGNED_COMMITS"$'\n'"  - $SHORT_SHA: $MESSAGE"
            UNSIGNED_COUNT=$((UNSIGNED_COUNT + 1))
        fi
    fi
done

echo ""

if [ $UNSIGNED_COUNT -gt 0 ]; then
    echo -e "${RED}=========================================="
    echo "ERROR: Found $UNSIGNED_COUNT unsigned commit(s):"
    echo -e "$UNSIGNED_COMMITS"
    echo ""
    echo "All commits must be signed."
    echo ""
    echo "To sign your commits, you can:"
    echo "  1. Configure git to sign commits: git config commit.gpgsign true"
    echo "  2. Re-sign existing commits: git rebase -i $BASE_REF --exec \"git commit --amend --no-edit -S\""
    echo ""
    echo "See: https://docs.github.com/en/authentication/managing-commit-signature-verification"
    echo -e "==========================================${NC}"

    # Output for CI usage
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "unsigned_count=$UNSIGNED_COUNT" >> "$GITHUB_OUTPUT"
        {
            echo "unsigned_commits<<EOF"
            echo -e "$UNSIGNED_COMMITS"
            echo "EOF"
        } >> "$GITHUB_OUTPUT"
    fi

    exit 1
fi

echo -e "${GREEN}✅ All $COMMIT_COUNT commit(s) are properly signed!${NC}"
