"""
Test that solution notebooks execute without errors.
"""

import pytest
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Define the path to the notebooks directory
NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / 'notebooks'

# Discover all solution notebooks
solution_notebooks = sorted(NOTEBOOKS_DIR.rglob('*SOLUTION*.ipynb'))

# Create test IDs from notebook paths for better test output
notebook_ids = [nb.relative_to(NOTEBOOKS_DIR).as_posix() for nb in solution_notebooks]


@pytest.mark.parametrize('notebook_path', solution_notebooks, ids=notebook_ids)
def test_solution_notebook_executes(notebook_path):
    """
    Test that a solution notebook executes without errors.

    Uses nbclient to execute all cells in the notebook.
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Create a client to execute the notebook
    # timeout=600 means 10 minutes per cell
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name='python3',
        resources={'metadata': {'path': str(notebook_path.parent)}}
    )

    # Execute the notebook
    try:
        client.execute()
    except CellExecutionError as e:
        # Provide detailed error information
        pytest.fail(
            f"Notebook execution failed in cell {e.cell_index}:\n"
            f"{e.traceback}"
        )
    except Exception as e:
        # Catch any other execution errors
        pytest.fail(f"Notebook execution failed: {str(e)}")
