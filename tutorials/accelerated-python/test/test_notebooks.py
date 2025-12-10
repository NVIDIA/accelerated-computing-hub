"""
Test that solution notebooks execute without errors.
"""

import pytest
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Define the path to the notebooks directory
NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / 'notebooks'

# Discover all solution notebooks (excluding checkpoint files)
solution_notebooks = sorted([
    nb for nb in NOTEBOOKS_DIR.rglob('*SOLUTION*.ipynb')
    if '.ipynb_checkpoints' not in str(nb)
])

# Create test IDs from notebook paths for better test output
notebook_ids = [nb.relative_to(NOTEBOOKS_DIR).as_posix() for nb in solution_notebooks]


def extract_cell_outputs(nb, cell_times=None):
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
            time_str = f" ({cell_times.get(i, 0):.2f}s)" if cell_times else ""
            outputs.append(f"--- Cell {i}{time_str}: {source_preview}... ---\n" + ''.join(cell_outputs))
    return '\n'.join(outputs)


def check_gpu_state():
    """Print GPU state for debugging slow execution."""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_mode,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,utilization.gpu', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  GPU State: {result.stdout.strip()}")
        # Also check for any processes using the GPU
        result2 = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,name,used_memory', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result2.returncode == 0 and result2.stdout.strip():
            print(f"  GPU Processes: {result2.stdout.strip()}")
    except Exception as e:
        print(f"  GPU State check failed: {e}")


@pytest.mark.parametrize('notebook_path', solution_notebooks, ids=notebook_ids)
def test_solution_notebook_executes(notebook_path):
    """
    Test that a solution notebook executes without errors.

    Uses nbclient to execute all cells in the notebook.
    """
    print(f"\n=== Starting notebook: {notebook_path.name} ===")
    check_gpu_state()
    notebook_start = time.time()

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

    # Execute the notebook cell by cell to get timing
    cell_times = {}
    try:
        with client.setup_kernel():
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != 'code':
                    continue
                cell_start = time.time()
                source_preview = cell.source[:60].replace('\n', ' ')
                print(f"  Cell {i}: {source_preview}...", end='', flush=True)

                # Check kernel is alive before executing
                if not client.kc.is_alive():
                    print(" [KERNEL DEAD!]")
                    raise RuntimeError(f"Kernel died before cell {i}")

                client.execute_cell(cell, i)
                cell_time = time.time() - cell_start
                cell_times[i] = cell_time
                print(f" [{cell_time:.2f}s]")

                # Flush any pending output
                import sys
                sys.stdout.flush()

    except CellExecutionError as e:
        # Provide detailed error information
        # Include output from ALL cells, not just the failing one
        all_outputs = extract_cell_outputs(nb, cell_times)
        total_time = time.time() - notebook_start
        pytest.fail(f"Notebook execution failed (total time: {total_time:.2f}s):\n{str(e)}\n\n=== ALL CELL OUTPUTS ===\n{all_outputs}")
    except Exception as e:
        # Catch any other execution errors
        all_outputs = extract_cell_outputs(nb, cell_times)
        total_time = time.time() - notebook_start
        pytest.fail(f"Notebook execution failed (total time: {total_time:.2f}s): {str(e)}\n\n=== ALL CELL OUTPUTS ===\n{all_outputs}")

    total_time = time.time() - notebook_start
    print(f"=== Completed {notebook_path.name} in {total_time:.2f}s ===")
