"""
Test that the tutorial notebooks execute without errors.

The SWE ladder is sequential: notebooks 01 to 05 each append a row to
timings.json, and 06 reads them. Run the numbered notebooks in
order in one shared kernel state per notebook, and run 06 last so the json
exist.
"""

import time
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _runnable_notebook(stem):
    """Pick the solution variant if it exists"""
    sol = NOTEBOOKS_DIR / "solutions" / f"{stem}__SOLUTION.ipynb"
    if sol.exists():
        return sol
    matches = sorted(NOTEBOOKS_DIR.glob(f"{stem}*.ipynb"))
    return matches[0] if matches else None

LADDER_STEMS = [
    "01__swe_core__reference_solver",
    "02__jax",
    "03__pyomp",
    "04__nanobind",
    "05__cppjit",
    "06__synthesis",
]
ladder = [(stem, _runnable_notebook(stem)) for stem in LADDER_STEMS]
ladder = [(stem, nb) for stem, nb in ladder if nb is not None]
ladder_ids = [stem for stem, _ in ladder]


def _execute(notebook_path):
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    kernel_name = nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()


@pytest.mark.parametrize("stem,notebook_path", ladder, ids=ladder_ids)
def test_notebook_executes(stem, notebook_path):
    """Execute one ladder notebook"""
    start = time.time()
    try:
        _execute(notebook_path)
    except CellExecutionError as e:
        pytest.fail(f"{notebook_path.name} failed after {time.time() - start:.1f}s:\n{e}")
    print(f"{notebook_path.name} ran in {time.time() - start:.1f}s")
