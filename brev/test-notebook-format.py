#!/usr/bin/env python3
"""
Test Jupyter notebook format integrity and metadata.

This script performs three checks on every notebook:
  1. Structural integrity: validates the notebook against the official Jupyter
     notebook JSON schema using nbformat.
  2. Metadata conformance: verifies that the top-level metadata, nbformat, and
     nbformat_minor fields match the expected values.
  3. Clean outputs: non-SOLUTION notebooks must have all cell outputs,
     execution counts, and execution timing metadata cleared.

The cuDF kernelspec is accepted as an alternative to the default ipykernel.
If a notebook has any other kernelspec (or none), it is treated as incorrect
and --fix will replace it with the default.

Usage:
  ./brev/test-notebook-format.py                       # check all tutorials
  ./brev/test-notebook-format.py <tutorial-name>       # check one tutorial
  ./brev/test-notebook-format.py <tutorial-name> --fix # check and fix one tutorial
  ./brev/test-notebook-format.py --fix                 # check and fix all tutorials

Examples:
  ./brev/test-notebook-format.py
  ./brev/test-notebook-format.py accelerated-python
  ./brev/test-notebook-format.py cuda-cpp --fix
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import nbformat

# Standard metadata expected for all notebooks.
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

# The cuDF kernelspec is accepted as an alternative to the default.
CUDF_KERNELSPEC = {
    "display_name": "Python 3 (RAPIDS 25.10)",
    "language": "python",
    "name": "cudf-cu13-25.10",
}

# ANSI colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"  # No Color


def has_cudf_kernelspec(metadata: dict) -> bool:
    """Check if a notebook's metadata contains the cuDF kernelspec."""
    return metadata.get("kernelspec") == CUDF_KERNELSPEC


def get_expected_metadata(metadata: dict) -> dict:
    """
    Return the expected metadata dict for a notebook.

    If the notebook already has the cuDF kernelspec, it is preserved.
    Otherwise the default kernelspec is expected.
    """
    expected = dict(STANDARD_METADATA)
    if has_cudf_kernelspec(metadata):
        expected = dict(expected)
        expected["kernelspec"] = dict(CUDF_KERNELSPEC)
    return expected


def diff_metadata(actual: dict, expected: dict, path: str = "") -> list[str]:
    """
    Recursively compare actual metadata against expected metadata.

    Returns a list of human-readable difference descriptions.
    """
    diffs = []
    prefix = f"{path}." if path else ""

    # Check for missing keys
    for key in expected:
        if key not in actual:
            diffs.append(f"  Missing key: {prefix}{key}")
        elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
            diffs.extend(diff_metadata(actual[key], expected[key], f"{prefix}{key}"))
        elif actual[key] != expected[key]:
            diffs.append(
                f"  Wrong value for {prefix}{key}: "
                f"got {json.dumps(actual[key])}, "
                f"expected {json.dumps(expected[key])}"
            )

    # Check for extra keys
    for key in actual:
        if key not in expected:
            diffs.append(f"  Extra key: {prefix}{key}")

    return diffs


def is_solution_notebook(notebook_path: Path) -> bool:
    """Check if a notebook is a SOLUTION notebook (filename contains SOLUTION)."""
    return "SOLUTION" in notebook_path.name


def check_clean_outputs(notebook: dict) -> list[str]:
    """
    Check that code cells have no outputs, execution counts, or execution
    timing metadata (from the jupyterlab-execute-time plugin).

    Returns a list of problem descriptions (empty if clean).
    """
    problems = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            problems.append(f"  Cell {i} has non-empty outputs")
        if cell.get("execution_count") is not None:
            problems.append(f"  Cell {i} has execution_count={cell['execution_count']}")
        if "execution" in cell.get("metadata", {}):
            problems.append(f"  Cell {i} has execution timing metadata")
    return problems


def strip_outputs(notebook: dict) -> None:
    """Clear outputs, execution counts, and execution timing metadata from all
    code cells in-place."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)


def validate_notebook_schema(notebook_path: Path) -> list[str]:
    """
    Validate a notebook against the official Jupyter notebook JSON schema.

    Returns a list of validation error messages (empty if valid).
    """
    errors = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nb = nbformat.read(str(notebook_path), as_version=4)
            nbformat.validate(nb)
    except nbformat.ValidationError as e:
        errors.append(f"  Schema validation error: {e.message}")
    except Exception as e:
        errors.append(f"  Failed to read notebook: {e}")
    return errors


def check_notebook(notebook_path: Path, fix: bool) -> bool:
    """
    Check a single notebook's format and metadata.

    Returns True if the notebook passes all checks.
    If fix=True, corrects the metadata in-place.
    """
    # Phase 1: Validate notebook structure against the JSON schema.
    schema_errors = validate_notebook_schema(notebook_path)

    # Phase 2: Check metadata conformance.
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"{RED}✗{NC} {notebook_path}")
        print(f"  Failed to read notebook: {e}")
        return False

    actual_metadata = notebook.get("metadata", {})
    expected_metadata = get_expected_metadata(actual_metadata)
    actual_nbformat = notebook.get("nbformat")
    actual_nbformat_minor = notebook.get("nbformat_minor")

    tag = "cudf" if has_cudf_kernelspec(actual_metadata) else "standard"

    problems = []

    # Add schema errors
    problems.extend(schema_errors)

    # Check metadata
    metadata_diffs = diff_metadata(actual_metadata, expected_metadata, "metadata")
    if metadata_diffs:
        problems.extend(metadata_diffs)

    # Check nbformat
    if actual_nbformat != STANDARD_NBFORMAT:
        problems.append(
            f"  Wrong nbformat: got {actual_nbformat}, expected {STANDARD_NBFORMAT}"
        )

    # Check nbformat_minor
    if actual_nbformat_minor != STANDARD_NBFORMAT_MINOR:
        problems.append(
            f"  Wrong nbformat_minor: got {actual_nbformat_minor}, "
            f"expected {STANDARD_NBFORMAT_MINOR}"
        )

    # Phase 3: Non-SOLUTION notebooks must have clean outputs.
    if not is_solution_notebook(notebook_path):
        output_problems = check_clean_outputs(notebook)
        if output_problems:
            problems.extend(output_problems)

    if not problems:
        print(f"{GREEN}✓{NC} {notebook_path} ({tag})")
        return True

    # There are problems
    print(f"{RED}✗{NC} {notebook_path} ({tag})")
    for problem in problems:
        print(f"  {problem}")

    if fix:
        notebook["metadata"] = expected_metadata
        notebook["nbformat"] = STANDARD_NBFORMAT
        notebook["nbformat_minor"] = STANDARD_NBFORMAT_MINOR

        if not is_solution_notebook(notebook_path):
            strip_outputs(notebook)

        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write("\n")

        print(f"  {GREEN}→ Fixed{NC}")

    return False


def find_notebook_dirs(repo_root: Path) -> list[Path]:
    """
    Return all directories that should be checked for notebooks.

    This includes every subdirectory under tutorials/ and the
    Accelerated_Python_User_Guide directory (if it exists).
    """
    dirs = []

    tutorials_root = repo_root / "tutorials"
    if tutorials_root.is_dir():
        dirs.extend(sorted(d for d in tutorials_root.iterdir() if d.is_dir()))

    user_guide = repo_root / "Accelerated_Python_User_Guide"
    if user_guide.is_dir():
        dirs.append(user_guide)

    return dirs


def resolve_tutorial_path(tutorial_arg: str, repo_root: Path) -> Path:
    """Resolve a tutorial argument to an absolute directory path."""
    if "/" in tutorial_arg or Path(tutorial_arg).is_dir():
        path = Path(tutorial_arg)
        if not path.is_absolute():
            path = repo_root / path
        return path
    return repo_root / "tutorials" / tutorial_arg


def check_directory(dir_path: Path, repo_root: Path, fix: bool) -> tuple[int, int]:
    """
    Check all notebooks in a directory.

    Returns (passed, failed) counts.
    """
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


def main():
    parser = argparse.ArgumentParser(
        description="Test Jupyter notebook format integrity and metadata."
    )
    parser.add_argument(
        "tutorial",
        nargs="?",
        default=None,
        help=(
            'Tutorial name (e.g., "accelerated-python") or path to tutorial '
            "directory. If omitted, all tutorials and the "
            "Accelerated_Python_User_Guide are checked."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically correct metadata that does not match",
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.fix:
        print(f"{YELLOW}Fix mode enabled: metadata will be corrected in-place{NC}")
        print()

    # Determine which directories to check
    if args.tutorial is not None:
        tutorial_path = resolve_tutorial_path(args.tutorial, repo_root)
        if not tutorial_path.is_dir():
            print(f"{RED}Error: Tutorial directory not found: {tutorial_path}{NC}")
            sys.exit(1)
        dirs_to_check = [tutorial_path]
    else:
        dirs_to_check = find_notebook_dirs(repo_root)
        if not dirs_to_check:
            print(f"{YELLOW}No tutorial directories found in {repo_root}{NC}")
            sys.exit(0)

    total_passed = 0
    total_failed = 0

    for dir_path in dirs_to_check:
        passed, failed = check_directory(dir_path, repo_root, args.fix)
        total_passed += passed
        total_failed += failed

    print("=" * 80)
    if total_failed == 0:
        print(
            f"{GREEN}✅ All {total_passed} notebook(s) have correct format and "
            f"metadata!{NC}"
        )
        return 0
    else:
        action = "fixed" if args.fix else "failed"
        print(
            f"{RED}❌ {total_failed} notebook(s) {action}, "
            f"{total_passed} passed out of {total_passed + total_failed} total{NC}"
        )
        if not args.fix:
            print(f"\nRun with --fix to automatically correct metadata.")
        return 0 if args.fix else 1


if __name__ == "__main__":
    sys.exit(main())
