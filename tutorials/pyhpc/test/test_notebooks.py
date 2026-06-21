"""
Test that the tutorial notebooks execute without errors.

The notebooks run as an ordered ladder (00 to 13). For each rung we prefer
the filled-in solution notebook when one exists, and otherwise fall back to
the exercise notebook (the intro/reference notebook 04, the synthesis
notebook 09, and the mpi4py walkthrough 03 have no separate solution and are
complete as written).

Ordering matters for the SWE sub-ladder: notebooks 04 to 08 each append a
row to timings.json and 09 reads them, so 04-08 must run before 09. pytest
executes the parametrized cases in list order, so listing the rungs 00..13
in order is sufficient.
"""

import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _runnable_notebook(stem):
    """Pick the solution variant if it exists, else the exercise notebook."""
    sol = NOTEBOOKS_DIR / "solutions" / f"{stem}__SOLUTION.ipynb"
    if sol.exists():
        return sol
    matches = sorted(NOTEBOOKS_DIR.glob(f"{stem}*.ipynb"))
    return matches[0] if matches else None


LADDER_STEMS = [
    "00__numpy",
    "01__cupy",
    "02__power_iteration__cupy__memory_spaces",
    "03__mpi4py",
    "04__swe__intro",
    "05__swe__jax",
    "06__swe__pyomp",
    "07__swe__nanobind",
    "08__swe__cppjit__thrust",
    "09__swe__synthesis",
    "10__power_iteraiton__cupy__asynchrony",
    "11__copy__kernel_authoring",
    "12__book_histogram__kernel_authoring",
    "13__cpp_interop",
]
ladder = [(stem, _runnable_notebook(stem)) for stem in LADDER_STEMS]
ladder = [(stem, nb) for stem, nb in ladder if nb is not None]
ladder_ids = [stem for stem, _ in ladder]


def _gpu_state():
    """One-line GPU snapshot for debugging slow/failed execution."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            print(f"  GPU: {out.stdout.strip()}")
    except Exception as e:  # noqa: BLE001 - debug aid only
        print(f"  GPU state check failed: {e}")


def _execute(notebook_path):
    """Execute a notebook cell-by-cell, printing per-cell timing."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    kernel_name = nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(
        nb,
        timeout=900,  # seconds per cell
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            preview = cell.source[:60].replace("\n", " ")
            print(f"  cell {i}: {preview}...", end="", flush=True)
            cell_start = time.time()
            client.execute_cell(cell, i)
            print(f" [{time.time() - cell_start:.1f}s]")
            sys.stdout.flush()


@pytest.mark.parametrize("stem,notebook_path", ladder, ids=ladder_ids)
def test_notebook_executes(stem, notebook_path):
    """Execute one ladder notebook and fail if any cell raises."""
    print(f"\n=== {notebook_path.relative_to(NOTEBOOKS_DIR)} ===")
    _gpu_state()
    start = time.time()
    try:
        _execute(notebook_path)
    except CellExecutionError as e:
        pytest.fail(f"{notebook_path.name} failed after {time.time() - start:.1f}s:\n{e}")
    print(f"{notebook_path.name} ran in {time.time() - start:.1f}s")
