# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation
# SPDX-License-Identifier: MIT
"""Source checkout and locked environment setup for the Article 2 notebooks."""
from pathlib import Path
import hashlib
import importlib.util
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
    for directory in (tutorial, lesson):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return lesson


def open_notebook(part: str, *, source_commit: str | None = None, benchmark: bool = False) -> dict:
    """Prepare a local checkout or the exact commit that supplied this helper."""
    try:
        hosted = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        hosted = False
    if source_commit is not None:
        repo_root = source_checkout(Path("/content/ach-mujoco-source"), source_commit)
    else:
        repo_root = Path(__file__).resolve().parents[4]
        if hosted and "HUB_SOURCE_REF" in os.environ:
            # Re-running a Colab cell now finds this helper in the checkout.
            # Still honor an explicitly changed ref, preserving existing files.
            ref = os.environ["HUB_SOURCE_REF"]
            refs = [ref] if len(ref) == 40 else subprocess.check_output(
                ["git", "ls-remote", REPO_URL, ref], text=True).splitlines()
            if len(refs) != 1:
                raise RuntimeError("HUB_SOURCE_REF must resolve to one upstream commit.")
            source_checkout(repo_root, refs[0].split()[0])
    tutorial = repo_root / TUTORIAL_RELATIVE
    revision = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True)
    print("Source:", revision.stdout.strip() if revision.returncode == 0 else "Image snapshot; retain source fingerprints.")
    if not benchmark:
        lesson = prepare_lesson(repo_root, part, hosted=hosted)
        print("Lesson directory:", lesson)
        return {"IN_COLAB": hosted, "tutorial_dir": tutorial, "lesson_dir": lesson}
    script = tutorial / part / "migration_benchmark.py"
    if not script.is_file():
        raise RuntimeError("This source revision does not contain migration_benchmark.py.")
    python = benchmark_python(tutorial, hosted=hosted)
    env = dict(os.environ)
    subprocess.run([str(python), "-c", "import sys; print(sys.executable); print(sys.version); "
                    "assert sys.version_info[:2] == (3, 12), 'Use the pinned Python 3.12 environment.'"],
                   check=True, env=env)
    print("Requirements lock SHA-256:", hashlib.sha256((tutorial / "requirements.lock.txt").read_bytes()).hexdigest())
    return {"IN_COLAB": hosted, "REPO_ROOT": repo_root, "TUTORIAL_DIR": tutorial,
            "PART2": tutorial / part, "SCRIPT": script, "BENCHMARK_PYTHON": python, "BENCHMARK_ENV": env}


def benchmark_python(tutorial: Path, *, hosted: bool) -> Path:
    """Use the selected local kernel or an isolated hash-locked Colab runtime."""
    if not hosted:
        return Path(sys.executable)
    runtime = Path("/content/mjwarp-benchmark-runtime")
    tools_env, environment = runtime / "tools", runtime / "python312"
    uv = tools_env / "bin/uv"
    if not uv.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(tools_env)], check=True)
        subprocess.run([str(tools_env / "bin/python"), "-m", "pip", "install", "uv==0.12.5"], check=True)
    python = environment / "bin/python"
    if not python.is_file():
        subprocess.run([str(uv), "venv", "--python", "3.12", str(environment)], check=True)
    subprocess.run([str(uv), "pip", "sync", "--python", str(python), "--require-hashes",
                    str(tutorial / "requirements.lock.txt")], check=True)
    return python
