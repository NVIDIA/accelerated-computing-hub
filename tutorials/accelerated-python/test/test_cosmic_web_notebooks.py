"""Test that the ordered cosmic-web sequence executes.

Applications 90 through 94 write per-stage timing files and application 95
reads them. Every notebook executes from the application directory so exercise
and solution variants use the same helpers and generated state. Pytest runs
parametrized cases in list order.
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
APPLICATION_DIR = NOTEBOOKS_DIR / "applications" / "cosmic-web"


def _runnable_notebook(stem):
    """Pick the solution variant if it exists, else the exercise notebook."""
    sol = APPLICATION_DIR / "solutions" / f"{stem}__SOLUTION.ipynb"
    if sol.exists():
        return sol
    matches = sorted(APPLICATION_DIR.glob(f"{stem}*.ipynb"))
    return matches[0] if matches else None


COSMIC_WEB_STEMS = [
    "90__cosmic_web__intro",
    "91__cosmic_web__cupy__array_model",
    "92__cosmic_web__jax__fusion",
    "93__cosmic_web__cccl__iterators",
    "94__cosmic_web__cppjit__cuda_kernel",
    "95__cosmic_web__synthesis",
]
sequence = [(stem, _runnable_notebook(stem)) for stem in COSMIC_WEB_STEMS]
missing = [stem for stem, notebook in sequence if notebook is None]
if missing:
    raise FileNotFoundError(f"Missing expected tutorial notebooks: {', '.join(missing)}")


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
    client = NotebookClient(
        nb,
        timeout=900,  # seconds per cell
        kernel_name="python3",
        resources={"metadata": {"path": str(APPLICATION_DIR)}},
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


@pytest.mark.parametrize("stem,notebook_path", sequence, ids=COSMIC_WEB_STEMS)
def test_notebook_executes(stem, notebook_path):
    """Execute one cosmic-web notebook and fail if any cell raises."""
    print(f"\n=== {notebook_path.relative_to(NOTEBOOKS_DIR)} ===")
    _gpu_state()
    start = time.time()
    try:
        _execute(notebook_path)
    except CellExecutionError as e:
        pytest.fail(f"{notebook_path.name} failed after {time.time() - start:.1f}s:\n{e}")
    print(f"{notebook_path.name} ran in {time.time() - start:.1f}s")
