#! /usr/bin/env python3
"""Launch small mpi4py examples from notebook 03.

The default notebook path uses Open MPI's mpirun. Prefer MPICH/Hydra when it is
installed and force Hydra's local fork launcher so it does not delegate to a
host srun mounted into the container.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def mpich_args(args: list[str]) -> list[str]:
    drop_value_options = {"-c", "--cpus-per-proc", "--map-by"}
    keep_value_options = {"-n", "-np", "--np", "--hosts", "-hosts"}
    cleaned = []
    i = 0
    before_program = True
    while i < len(args):
        # The notebook's Open MPI examples use "-c N" to request CPUs per
        # process. MPICH's local Hydra launcher does not accept that option.
        if before_program and args[i] in drop_value_options and i + 1 < len(args):
            i += 2
            continue
        if before_program and args[i] in keep_value_options and i + 1 < len(args):
            cleaned.extend(args[i : i + 2])
            i += 2
            continue
        if before_program and args[i].startswith("-"):
            raise ValueError(f"unsupported MPI launcher option for MPICH: {args[i]}")
        if before_program and not args[i].startswith("-"):
            before_program = False
        cleaned.append(args[i])
        i += 1
    executable = shutil.which("mpiexec.mpich")
    if not executable:
        raise RuntimeError("MPICH launcher requested, but mpiexec.mpich is unavailable")
    return [
        executable,
        "-launcher",
        "fork",
        *cleaned,
    ]


def openmpi_args(args: list[str], *, require_openmpi: bool = False) -> list[str]:
    executable = shutil.which("mpirun.openmpi")
    if not executable and not require_openmpi:
        executable = shutil.which("mpirun")
    if not executable:
        raise RuntimeError("Open MPI launcher requested, but mpirun.openmpi is unavailable")
    return [executable, "--oversubscribe", *args]


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: mpi4py_launcher.py -n <ranks> <program> [args...]", file=sys.stderr)
        return 2

    launcher = os.environ.get("ACH_MPI_LAUNCHER")
    if launcher not in (None, "mpich", "openmpi"):
        print(f"unsupported ACH_MPI_LAUNCHER: {launcher}", file=sys.stderr)
        return 2

    try:
        if launcher == "mpich" or (launcher is None and shutil.which("mpiexec.mpich")):
            # Keep the parent environment: stripping Slurm/PMI variables caused
            # mpi4py to hang during MPI_Init on Daint, even with Hydra's fork mode.
            command = mpich_args(args)
        else:
            command = openmpi_args(args, require_openmpi=launcher == "openmpi")
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
