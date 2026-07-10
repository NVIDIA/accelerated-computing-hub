from typing import Union, Literal  # to suppress irrelevant warnings from nbdistributed

from IPython.core.magic import register_cell_magic
from IPython import get_ipython

@register_cell_magic
def mgmn(line, cell):
    ipython = get_ipython()
    magics = ipython.magics_manager.magics["cell"]

    mpi_workers = False
    torch_distributed_workers = False

    if "px" in magics:
        view = magics["px"].__self__.view
        if view and len(view.client.ids) != 0:
            mpi_workers = True

    if "distributed" in magics and magics["distributed"].__self__._comm_manager is not None:
        torch_distributed_workers = True

    if mpi_workers and torch_distributed_workers:
        import logging
        logging.warning("Both the MPI and torch.distributed workers are running. Executing only on MPI workers.")

    if mpi_workers:
        ipython.run_cell_magic("px", "", cell)
    elif torch_distributed_workers:
        ipython.run_cell_magic("distributed", "", cell)
    else:
        print("No worker processes running.")
