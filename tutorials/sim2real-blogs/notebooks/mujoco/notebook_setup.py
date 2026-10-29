# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Source checkout and locked environment setup for the Article 2 notebooks."""
from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/NVIDIA/accelerated-computing-hub.git"
TUTORIAL_RELATIVE = Path("tutorials/sim2real-blogs/notebooks/mujoco")


def source_checkout(checkout: Path, revision: str) -> Path:
    """Fetch an exact upstream commit without replacing existing user files."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Resolve the source ref to a full commit before checkout.")
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not checkout.exists():
        checkout.mkdir(parents=True)
        subprocess.run(["git", "init", str(checkout)], check=True, env=env)
        subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", REPO_URL], check=True, env=env)
        subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "init", "--cone"], check=True, env=env)
        subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", str(TUTORIAL_RELATIVE)], check=True, env=env)
    remote = subprocess.check_output(["git", "-C", str(checkout), "remote", "get-url", "origin"], text=True).strip()
    if remote != REPO_URL:
        raise RuntimeError("Existing checkout uses another repository; choose a fresh COLAB_CHECKOUT.")
    existing = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    if existing.returncode == 0 and existing.stdout.strip() != revision:
        raise RuntimeError("Source changed; existing files were preserved. Choose a fresh COLAB_CHECKOUT.")
    if existing.returncode != 0:
        subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", revision], check=True, env=env)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", revision], check=True, env=env)
    return checkout


def prepare_lesson(repo_root: Path, part: str, *, hosted: bool = False) -> Path:
    """Select the lesson and install its lock only in a clean hosted kernel."""
    tutorial = repo_root / TUTORIAL_RELATIVE
    lesson = tutorial / part
    if not (lesson / "viewer_window.py").is_file():
        raise RuntimeError("This checkout does not contain the current box lessons.")
    if hosted:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("These teaching cells require a Python 3.12 kernel. Use the local/Brev environment; the benchmark notebook uses an isolated runtime.")
        if any(name in sys.modules for name in ("mujoco", "mujoco_warp", "warp")):
            raise RuntimeError("Restart the runtime before installing the locked physics packages.")
        os.environ.setdefault("MUJOCO_GL", "egl")
        subprocess.run([sys.executable, "-m", "pip", "install", "--require-hashes", "-r", str(tutorial / "requirements.lock.txt")], check=True)
    os.chdir(lesson)
    if str(lesson) not in sys.path:
        sys.path.insert(0, str(lesson))
    return lesson
