#!/usr/bin/env python3
"""
Test Jupyter notebook format integrity and metadata.

This script checks that every notebook is in **canonical format**: the exact
byte-for-byte output that nbformat.write() produces after applying our
metadata policy.  Canonical format means:

  - 1-space JSON indentation, sorted keys, ``(",", ": ")`` separators
    (the nbformat standard serialization).
  - Every cell has an ``id`` field (nbformat 4.5 requirement).
  - Top-level metadata matches the project standard (kernelspec, colab,
    language_info, etc.).
  - Non-SOLUTION notebooks have clean outputs (no outputs, execution
    counts, or execution timing metadata).
  - SOLUTION notebook outputs are well-formed (e.g. stream outputs have
    the required ``name`` field).
  - nbformat 4, nbformat_minor 5.

If a notebook has any non-standard kernelspec (or none), --fix replaces it
with the default.

Usage:
  ./brev/test-notebook-format.py                       # check all tutorials
  ./brev/test-notebook-format.py <tutorial-name>       # check one tutorial
  ./brev/test-notebook-format.py <tutorial-name> --fix # check and fix one tutorial
  ./brev/test-notebook-format.py --fix                 # check and fix all tutorials
  ./brev/test-notebook-format.py a.ipynb b.ipynb       # check specific files
  ./brev/test-notebook-format.py a.ipynb --fix         # fix specific files

Examples:
  ./brev/test-notebook-format.py
  ./brev/test-notebook-format.py accelerated-python
  ./brev/test-notebook-format.py cuda-cpp --fix
  ./brev/test-notebook-format.py tutorials/accelerated-python/notebooks/start.ipynb
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import nbformat

# ---------------------------------------------------------------------------
# Metadata policy
# ---------------------------------------------------------------------------

STANDARD_METADATA = {
    "accelerator": "GPU",
    "colab": {
        "gpuType": "T4",
        "provenance": [],
        "toc_visible": True,
    },
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3,
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.7",
    },
}

STANDARD_NBFORMAT = 4
STANDARD_NBFORMAT_MINOR = 5

# ANSI colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"  # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_solution_notebook(notebook_path: Path) -> bool:
    return "SOLUTION" in notebook_path.name


def diff_metadata(actual: dict, expected: dict, path: str = "") -> list[str]:
    """Recursively compare metadata.  Returns human-readable differences."""
    diffs: list[str] = []
    prefix = f"{path}." if path else ""

    for key in expected:
        if key not in actual:
            diffs.append(f"  Missing key: {prefix}{key}")
        elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
            diffs.extend(
                diff_metadata(actual[key], expected[key], f"{prefix}{key}")
            )
        elif actual[key] != expected[key]:
            diffs.append(
                f"  Wrong value for {prefix}{key}: "
                f"got {json.dumps(actual[key])}, "
                f"expected {json.dumps(expected[key])}"
            )

    for key in actual:
        if key not in expected:
            diffs.append(f"  Extra key: {prefix}{key}")

    return diffs


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def canonicalize_notebook(notebook_path: Path) -> tuple[str, list[str]]:
    """Read a notebook and return ``(canonical_text, problems)``.

    *canonical_text* is the byte-for-byte content the file should have.
    *problems* is a list of human-readable descriptions of anything that
    had to be changed to reach canonical form (empty when the file is
    already canonical).
    """
    problems: list[str] = []
    solution = is_solution_notebook(notebook_path)

    # -- Read raw JSON to inspect the original state -------------------------
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except OSError as e:
        return "", [f"  Failed to read file: {e}"]

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return "", [f"  Invalid JSON: {e}"]

    # -- Detect original metadata problems before nbformat.read() -----------
    actual_metadata = raw.get("metadata", {})
    expected_metadata = dict(STANDARD_METADATA)
    metadata_diffs = diff_metadata(actual_metadata, expected_metadata, "metadata")
    if metadata_diffs:
        problems.extend(metadata_diffs)

    if raw.get("nbformat") != STANDARD_NBFORMAT:
        problems.append(
            f"  Wrong nbformat: got {raw.get('nbformat')}, "
            f"expected {STANDARD_NBFORMAT}"
        )
    if raw.get("nbformat_minor") != STANDARD_NBFORMAT_MINOR:
        problems.append(
            f"  Wrong nbformat_minor: got {raw.get('nbformat_minor')}, "
            f"expected {STANDARD_NBFORMAT_MINOR}"
        )

    # Check outputs for non-SOLUTION notebooks
    if not solution:
        for i, cell in enumerate(raw.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs"):
                problems.append(f"  Cell {i} has non-empty outputs")
            if cell.get("execution_count") is not None:
                problems.append(
                    f"  Cell {i} has execution_count={cell['execution_count']}"
                )
            if "execution" in cell.get("metadata", {}):
                problems.append(f"  Cell {i} has execution timing metadata")

    # Check for missing cell IDs
    cells_missing_ids = sum(
        1 for c in raw.get("cells", []) if "id" not in c
    )
    if cells_missing_ids:
        problems.append(f"  {cells_missing_ids} cell(s) missing id field")

    # Check for malformed outputs (SOLUTION notebooks keep outputs)
    if solution:
        for i, cell in enumerate(raw.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            for j, out in enumerate(cell.get("outputs", [])):
                otype = out.get("output_type")
                if otype == "stream" and "name" not in out:
                    problems.append(
                        f"  Cell {i} output {j}: stream output missing 'name'"
                    )
                if otype in ("execute_result", "display_data"):
                    if "metadata" not in out:
                        problems.append(
                            f"  Cell {i} output {j}: {otype} missing 'metadata'"
                        )
                if otype == "execute_result" and "execution_count" not in out:
                    problems.append(
                        f"  Cell {i} output {j}: execute_result missing 'execution_count'"
                    )

    # -- Build canonical form via nbformat -----------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = nbformat.read(str(notebook_path), as_version=4)

    # Apply metadata policy
    nb.metadata = nbformat.from_dict(expected_metadata)
    nb.nbformat = STANDARD_NBFORMAT
    nb.nbformat_minor = STANDARD_NBFORMAT_MINOR

    # Fix malformed outputs
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            otype = out.get("output_type")
            if otype == "stream" and "name" not in out:
                out["name"] = "stdout"
            if otype in ("execute_result", "display_data") and "metadata" not in out:
                out["metadata"] = {}
            if otype == "execute_result" and "execution_count" not in out:
                out["execution_count"] = cell.get("execution_count")

    # Strip outputs for non-SOLUTION notebooks
    if not solution:
        for cell in nb.cells:
            if cell.get("cell_type") != "code":
                continue
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.get("metadata", {}).pop("execution", None)

    # Serialize canonically (nbformat uses indent=1, sort_keys=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        canonical_text = nbformat.writes(nb)
    if not canonical_text.endswith("\n"):
        canonical_text += "\n"

    # -- Detect format-only problems ----------------------------------------
    if not problems and raw_text != canonical_text:
        # Content is semantically correct but format differs (indentation,
        # key ordering, missing cell IDs, etc.)
        raw_lines = raw_text.split("\n")
        indent = 0
        for line in raw_lines[1:5]:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                break
        if indent != 1:
            problems.append(
                f"  Non-canonical indentation ({indent}-space, expected 1-space)"
            )

        # Key ordering
        for i, cell in enumerate(raw.get("cells", [])):
            if list(cell.keys()) != sorted(cell.keys()):
                problems.append(f"  Cell {i}: keys not in canonical order")
                break

        if not problems:
            problems.append("  File not in canonical format (run --fix)")

    return canonical_text, problems


# ---------------------------------------------------------------------------
# Check / fix
# ---------------------------------------------------------------------------


def check_notebook(notebook_path: Path, fix: bool) -> bool:
    """Check a single notebook.  Returns True if it passes."""
    canonical_text, problems = canonicalize_notebook(notebook_path)

    if not canonical_text and problems:
        print(f"{RED}✗{NC} {notebook_path}")
        for p in problems:
            print(p)
        return False

    # Read the original to compare byte-for-byte
    with open(notebook_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    if original_text == canonical_text:
        print(f"{GREEN}✓{NC} {notebook_path}")
        return True

    # There are problems
    print(f"{RED}✗{NC} {notebook_path}")
    for p in problems:
        print(p)

    if fix:
        with open(notebook_path, "w", encoding="utf-8") as f:
            f.write(canonical_text)
        print(f"  {GREEN}→ Fixed{NC}")

    return False


# ---------------------------------------------------------------------------
# Directory / CLI plumbing
# ---------------------------------------------------------------------------


def find_notebook_dirs(repo_root: Path) -> list[Path]:
    dirs = []
    tutorials_root = repo_root / "tutorials"
    if tutorials_root.is_dir():
        dirs.extend(sorted(d for d in tutorials_root.iterdir() if d.is_dir()))
    user_guide = repo_root / "Accelerated_Python_User_Guide"
    if user_guide.is_dir():
        dirs.append(user_guide)
    return dirs


def resolve_tutorial_path(tutorial_arg: str, repo_root: Path) -> Path:
    if "/" in tutorial_arg or Path(tutorial_arg).is_dir():
        path = Path(tutorial_arg)
        if not path.is_absolute():
            path = repo_root / path
        return path
    return repo_root / "tutorials" / tutorial_arg


def check_directory(dir_path: Path, repo_root: Path, fix: bool) -> tuple[int, int]:
    notebooks = sorted(dir_path.rglob("*.ipynb"))
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]

    if not notebooks:
        return 0, 0

    try:
        display_path = dir_path.relative_to(repo_root)
    except ValueError:
        display_path = dir_path
    print(f"Checking notebook format in: {display_path}")
    print()

    passed = 0
    failed = 0

    for notebook_path in notebooks:
        if check_notebook(notebook_path, fix):
            passed += 1
        else:
            failed += 1

    print()
    return passed, failed


def check_files(notebook_paths: list[Path], fix: bool) -> tuple[int, int]:
    """Check a list of individual notebook files.

    Returns (passed, failed) counts.
    """
    passed = 0
    failed = 0
    for notebook_path in notebook_paths:
        if check_notebook(notebook_path, fix):
            passed += 1
        else:
            failed += 1
    return passed, failed


def main():
    parser = argparse.ArgumentParser(
        description="Test Jupyter notebook format integrity and metadata."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Notebook files (.ipynb), a tutorial name, or a directory.  "
            "If omitted, all tutorials and the Accelerated_Python_User_Guide "
            "are checked."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically correct formatting, metadata, and outputs",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.fix:
        print(f"{YELLOW}Fix mode enabled: notebooks will be rewritten in canonical format{NC}")
        print()

    total_passed = 0
    total_failed = 0

    if args.paths:
        # If any path ends in .ipynb, treat all args as individual files.
        notebook_files = [Path(p) for p in args.paths if p.endswith(".ipynb")]
        if notebook_files:
            # Filter to only .ipynb args (ignore any non-.ipynb args).
            passed, failed = check_files(notebook_files, args.fix)
            total_passed += passed
            total_failed += failed
        else:
            # Single tutorial name / directory path (legacy behavior).
            tutorial_path = resolve_tutorial_path(args.paths[0], repo_root)
            if not tutorial_path.is_dir():
                print(f"{RED}Error: Tutorial directory not found: {tutorial_path}{NC}")
                sys.exit(1)
            passed, failed = check_directory(tutorial_path, repo_root, args.fix)
            total_passed += passed
            total_failed += failed
    else:
        dirs_to_check = find_notebook_dirs(repo_root)
        if not dirs_to_check:
            print(f"{YELLOW}No tutorial directories found in {repo_root}{NC}")
            sys.exit(0)
        for dir_path in dirs_to_check:
            passed, failed = check_directory(dir_path, repo_root, args.fix)
            total_passed += passed
            total_failed += failed

    print("=" * 80)
    if total_failed == 0:
        print(
            f"{GREEN}✅ All {total_passed} notebook(s) are in canonical format!{NC}"
        )
        return 0
    else:
        action = "fixed" if args.fix else "failed"
        print(
            f"{RED}❌ {total_failed} notebook(s) {action}, "
            f"{total_passed} passed out of {total_passed + total_failed} total{NC}"
        )
        if not args.fix:
            print(f"\nRun with --fix to automatically rewrite to canonical format.")
        return 0 if args.fix else 1


if __name__ == "__main__":
    sys.exit(main())
