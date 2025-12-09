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


def extract_cell_outputs(nb):
    """Extract stdout/stderr from all executed cells for debugging."""
    outputs = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        cell_outputs = []
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stream':
                stream_name = output.get('name', 'stdout')
                text = output.get('text', '')
                cell_outputs.append(f"[{stream_name}] {text}")
            elif output.get('output_type') == 'error':
                ename = output.get('ename', 'Error')
                evalue = output.get('evalue', '')
                cell_outputs.append(f"[error] {ename}: {evalue}")
        if cell_outputs:
            source_preview = cell.source[:100].replace('\n', ' ')
            outputs.append(f"--- Cell {i}: {source_preview}... ---\n" + ''.join(cell_outputs))
    return '\n'.join(outputs)


@pytest.mark.parametrize('notebook_path', solution_notebooks, ids=notebook_ids)
def test_solution_notebook_executes(notebook_path):
    """
    Test that a solution notebook executes without errors.

    Uses nbclient to execute all cells in the notebook.
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Determine which kernel to use
    # If the notebook specifies a kernel, use it; otherwise use python3
    kernel_name = 'python3'
    if 'kernelspec' in nb.metadata and 'name' in nb.metadata.kernelspec:
        kernel_name = nb.metadata.kernelspec.name

    # Create a client to execute the notebook
    # timeout=600 means 10 minutes per cell
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name=kernel_name,
        resources={'metadata': {'path': str(notebook_path.parent)}}
    )

    # Execute the notebook
    try:
        client.execute()
    except CellExecutionError as e:
        # Provide detailed error information
        # Include output from ALL cells, not just the failing one
        all_outputs = extract_cell_outputs(nb)
        pytest.fail(f"Notebook execution failed:\n{str(e)}\n\n=== ALL CELL OUTPUTS ===\n{all_outputs}")
    except Exception as e:
        # Catch any other execution errors
        all_outputs = extract_cell_outputs(nb)
        pytest.fail(f"Notebook execution failed: {str(e)}\n\n=== ALL CELL OUTPUTS ===\n{all_outputs}")
