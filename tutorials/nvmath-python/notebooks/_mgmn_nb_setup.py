from typing import Union, Literal  # to suppress irrelevant warnings from nbdistributed

from IPython.core.magic import register_cell_magic, needs_local_scope
from IPython import get_ipython

@register_cell_magic
@needs_local_scope
def mgmn(line, cell, local_ns=None):
    ipython = get_ipython()
    if local_ns and local_ns["using_mpi"]:
        ipython.run_cell_magic("px", "", cell)
    elif local_ns and local_ns["using_torch_distributed"]:
        ipython.run_cell_magic("distributed", "", cell)
    else:
        print("No worker processes running")

using_torch_distributed = False
using_mpi = False
