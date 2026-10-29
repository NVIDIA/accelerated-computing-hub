# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation
# SPDX-License-Identifier: MIT
"""Source checkout and isolated Python setup for the optional benchmark notebook."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

TUTORIAL_RELATIVE = Path("tutorials/newton/notebooks/newton")


def resolve_checkout(repo_url: str, revision: str, checkout: Path) -> Path:
    """Fetch one revision into a fresh checkout, preserving existing work."""
    checkout = Path(checkout)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not checkout.exists():
        checkout.mkdir(parents=True)
        subprocess.run(["git", "init", str(checkout)], check=True, env=env)
        subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", repo_url],
                       check=True, env=env)
        subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set",
                        str(TUTORIAL_RELATIVE)], check=True, env=env)
    actual_remote = subprocess.check_output(
        ["git", "-C", str(checkout), "remote", "get-url", "origin"], text=True).strip()
    if actual_remote != repo_url:
        raise RuntimeError("Existing checkout uses another repository; choose a fresh COLAB_CHECKOUT.")
    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True)
    if status.strip():
        raise RuntimeError("Existing checkout has edited tracked files; choose a fresh COLAB_CHECKOUT.")
    subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", revision],
                   check=True, env=env)
    requested = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "FETCH_HEAD"], text=True).strip()
    existing = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--verify", "HEAD"],
                              capture_output=True, text=True)
    if existing.returncode == 0 and existing.stdout.strip() != requested:
        raise RuntimeError("REF changed. Existing files were preserved; choose a fresh COLAB_CHECKOUT.")
    if existing.returncode != 0:
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", requested],
                       check=True, env=env)
    return checkout


def prepare_benchmark(*, hosted: bool, repo_url: str, revision: str,
                      checkout: Path, runtime_root: Path) -> tuple[Path, Path, dict[str, str]]:
    """Use the local kernel or a separate hash-locked hosted interpreter."""
    if hosted:
        root = resolve_checkout(repo_url, revision, checkout)
        tools_env = runtime_root / "tools"
        benchmark_env = runtime_root / "python312"
        uv = tools_env / "bin/uv"
        if not uv.is_file():
            subprocess.run([sys.executable, "-m", "venv", str(tools_env)], check=True)
            subprocess.run([str(tools_env / "bin/python"), "-m", "pip", "install", "uv==0.12.5"],
                           check=True)
        python = benchmark_env / "bin/python"
        if not python.is_file():
            subprocess.run([str(uv), "venv", "--python", "3.12", str(benchmark_env)], check=True)
        subprocess.run([str(uv), "pip", "sync", "--python", str(python), "--require-hashes",
                        str(root / TUTORIAL_RELATIVE / "requirements.lock.txt")], check=True)
    else:
        here = Path.cwd().resolve()
        root = next((p for p in (here, *here.parents)
                     if (p / TUTORIAL_RELATIVE / "requirements.lock.txt").is_file()), None)
        if root is None:
            raise RuntimeError("Open the notebook inside the accelerated-computing-hub checkout.")
        python = Path(sys.executable)
    tutorial = root / TUTORIAL_RELATIVE
    part3 = tutorial / "part3"
    if not (part3 / "migration_benchmark.py").is_file():
        raise RuntimeError("This revision does not contain migration_benchmark.py.")
    env = {**os.environ, "PXR_WORK_THREAD_LIMIT": "1"}
    subprocess.run([str(python), "-c",
                    "import sys; print(sys.executable); print(sys.version); "
                    "assert sys.version_info[:2] == (3, 12), 'Use the pinned Python 3.12 environment.'"],
                   check=True, env=env)
    source = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    print("Source:", source.stdout.strip() if source.returncode == 0 else
          "Image snapshot without Git metadata; retain the report's source fingerprints.")
    print("Tutorial:", tutorial)
    print("Requirements lock SHA-256:", hashlib.sha256((tutorial / "requirements.lock.txt").read_bytes()).hexdigest())
    return part3, python, env
