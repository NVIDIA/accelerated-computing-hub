#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check this machine for the tutorials and print the exact fix for each problem.

Run it with the Python of the environment you created in the Quickstart:

    python doctor.py              # checks the article whose packages are installed
    python doctor.py --article 3  # check a specific article
    python doctor.py --fetch      # also download the pinned robot models now

Only the standard library is required, so it also runs before anything is installed.
Exit code: 0 when nothing blocks the lessons, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PACKAGES = {
    "2": ("mujoco", "mujoco-warp", "warp-lang", "numpy", "glfw"),
    "3": ("newton", "warp-lang", "mujoco", "mujoco-warp", "usd-core", "numpy", "glfw"),
}
# Article folder per layout: this repository, or the accelerated-computing-hub copy.
_ARTICLE_DIRS = {"2": ("Article_2", "notebooks/mujoco"), "3": ("Article_3",)}
_PART = {"2": "part1", "3": "part3"}


def article_dir(article: str) -> str:
    """Article folder relative to REPO (the first layout that exists)."""
    for rel in _ARTICLE_DIRS[article]:
        if (REPO / rel).is_dir():
            return rel
    return _ARTICLE_DIRS[article][0]


PART_DIR = {a: REPO / article_dir(a) / _PART[a] for a in _ARTICLE_DIRS}


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, msg: str) -> None:
        print(f"[ OK ] {msg}")

    def warn(self, msg: str, fix: str | None = None) -> None:
        self.warnings += 1
        print(f"[WARN] {msg}")
        self._fix(fix)

    def fail(self, msg: str, fix: str) -> None:
        self.failures += 1
        print(f"[FAIL] {msg}")
        self._fix(fix)

    @staticmethod
    def _fix(fix: str | None) -> None:
        for i, line in enumerate((fix or "").splitlines()):
            print(f"       fix: {line}" if i == 0 else f"            {line}")


def install_command(article: str) -> str:
    rel = article_dir(article)
    venv = f"{rel}/.venv"
    return (
        f"cd {REPO}\n"
        f"uv venv --allow-existing --python 3.12 {venv}\n"
        f"uv pip install --python {venv}/bin/python --require-hashes -r {rel}/requirements.lock.txt\n"
        f"source {venv}/bin/activate"
    )


def pinned_versions(article: str) -> dict[str, str]:
    lock = REPO / article_dir(article) / "requirements.lock.txt"
    pins: dict[str, str] = {}
    if lock.is_file():
        for line in lock.read_text().splitlines():
            m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", line)
            if m:
                pins[m.group(1).lower()] = m.group(2)
    return pins


def installed(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def detect_article(requested: str | None) -> str:
    if requested:
        return requested
    return "3" if installed("newton") else "2"


def check_python(r: Report, article: str) -> None:
    v = sys.version_info
    where = f"{sys.executable} (Python {v.major}.{v.minor}.{v.micro})"
    if (v.major, v.minor) == (3, 12):
        r.ok(f"Python 3.12: {where}")
    else:
        r.fail(f"Python 3.12 is required; this is {where}", install_command(article))
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if not venv:
        r.warn("No virtual environment is active (packages would go into the system Python).",
               f"source {REPO}/{article_dir(article)}/.venv/bin/activate")


def check_packages(r: Report, article: str) -> bool:
    pins = pinned_versions(article)
    missing, mismatched = [], []
    for name in PACKAGES[article]:
        have, want = installed(name), pins.get(name)
        if have is None:
            missing.append(name)
        elif want and have != want:
            mismatched.append(f"{name} {have} (tested: {want})")
        else:
            r.ok(f"{name} {have}")
    if missing:
        fix = ("the environment setup above installs them" if r.failures
               else install_command(article))
        r.fail(f"Article {article} packages missing in this Python: {', '.join(missing)}", fix)
        return False
    if mismatched:
        r.warn("Versions differ from the tested lock: " + "; ".join(mismatched),
               install_command(article))
    if article == "2" and installed("newton"):
        r.warn("Newton is installed in the Article 2 environment; the two articles pin different "
               "MuJoCo/Warp versions.", f"use a separate environment per article:\n{install_command('2')}")
    return True


def check_tools(r: Report) -> None:
    if shutil.which("git"):
        r.ok("git is installed (used once to download the pinned robot models)")
    else:
        r.fail("git is not installed; the robot models are downloaded with git.",
               "Ubuntu/Debian: sudo apt install git    conda: conda install git    macOS: xcode-select --install")


def check_gpu(r: Report, article: str) -> None:
    smi = shutil.which("nvidia-smi")
    gpus = []
    if smi:
        try:
            out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=20)
            gpus = [line for line in out.stdout.splitlines() if line.startswith("GPU")]
        except (OSError, subprocess.TimeoutExpired):
            gpus = []
    if not gpus:
        r.warn("No NVIDIA GPU detected (nvidia-smi). Article 2 Part 1 runs on CPU; Part 2 and the "
               "GPU benchmarks need CUDA.",
               "install the NVIDIA driver (`nvidia-smi` must list a GPU), or stay on the CPU lessons")
        return
    r.ok(f"NVIDIA driver: {len(gpus)} GPU(s): " + "; ".join(g.split(" (UUID")[0] for g in gpus))
    if installed("warp-lang") is None:
        return
    code = ("import warp as wp; wp.config.quiet = True; wp.init(); "
            "print(wp.get_cuda_device_count())")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        count = int(out.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        count = -1
    if count > 0:
        r.ok(f"Warp sees {count} CUDA device(s): use --device cuda:0")
    else:
        r.warn("Warp cannot use the GPU (CUDA driver too old or not visible to this Python).",
               "update the NVIDIA driver (Warp needs CUDA 12 drivers), then rerun this check")


def check_display(r: Report) -> None:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        r.warn("No display (DISPLAY/WAYLAND_DISPLAY unset): viewer windows cannot open here.",
               "add --headless-steps 2000 (Article 2) or --viewer null (Article 3) to the lesson commands")
    else:
        r.ok("Display available for the viewer windows" if sys.platform.startswith("linux")
             else f"Platform {platform.system()}: viewer uses the native window system")


def check_article3_env(r: Report) -> None:
    if os.environ.get("PXR_WORK_THREAD_LIMIT") == "1":
        r.ok("PXR_WORK_THREAD_LIMIT=1 (required by usd-core 26.3)")
    else:
        r.fail("PXR_WORK_THREAD_LIMIT=1 is not set (usd-core 26.3 needs it before Python starts).",
               "export PXR_WORK_THREAD_LIMIT=1   # then restart Python/Jupyter")


def check_assets(r: Report, article: str, fetch: bool) -> None:
    sys.path.insert(0, str(PART_DIR[article]))
    try:
        import robots  # noqa: PLC0415
        import utils  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        r.warn(f"Cannot inspect robot models ({exc}).")
        return
    if not hasattr(utils, "find_cached_robot"):
        r.fail("These lesson files predate the automatic robot-model cache fix (13 Oct 2026).",
               f"cd {REPO} && git pull")
        return
    for env_name in ("MUJOCO_MENAGERIE_PATH", "NEWTON_MENAGERIE_PATH"):
        value = os.environ.get(env_name, "").strip()
        if value and not Path(value).expanduser().is_dir():
            r.warn(f"{env_name}={value} does not exist; it is ignored (models download automatically).",
                   f"unset {env_name}")
    missing = []
    for key in ("so101", "rebot"):
        spec = robots.get_robot(key)
        found = utils.find_cached_robot(spec)
        if found:
            r.ok(f"{spec.display_name} model cached: {found}")
        elif fetch:
            try:
                path = utils.download_robot_sparse(spec, utils.default_cache_dir(spec))
                r.ok(f"{spec.display_name} model downloaded: {path}")
            except Exception as exc:  # noqa: BLE001
                r.fail(f"{spec.display_name} model download failed.", str(exc))
        else:
            missing.append(spec)
    if not missing:
        return
    names = ", ".join(s.display_name for s in missing)
    url = missing[0].menagerie_url
    try:
        probe = subprocess.run(["git", "ls-remote", "--heads", url, "main"], capture_output=True,
                               text=True, timeout=30, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        reachable = probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        reachable = False
    target = utils.default_cache_dir(missing[0])
    if reachable:
        r.ok(f"Not downloaded yet: {names}. github.com is reachable, so the first lesson run "
             "downloads it automatically (or run: python doctor.py --fetch).")
    else:
        r.fail(f"Not downloaded yet: {names}, and {url} is not reachable from this machine.",
               "connect to the internet (check VPN/proxy), then: python doctor.py --fetch\n"
               f"offline: copy {target.parent} from a machine where the lessons already ran")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--article", choices=tuple(a for a in _ARTICLE_DIRS if (REPO / article_dir(a)).is_dir()),
                        help="default: detect from installed packages")
    parser.add_argument("--fetch", action="store_true", help="download missing robot models now")
    args = parser.parse_args()
    article = detect_article(args.article)
    print(f"Checking setup for Article {article} ({REPO})\n")
    r = Report()
    check_python(r, article)
    have_packages = check_packages(r, article)
    check_tools(r)
    check_gpu(r, article)
    check_display(r)
    if article == "3":
        check_article3_env(r)
    if have_packages:
        check_assets(r, article, args.fetch)
    print()
    if r.failures:
        print(f"{r.failures} problem(s) to fix (see 'fix:' lines above), {r.warnings} warning(s).")
        return 1
    print(f"Ready: no blocking problems ({r.warnings} warning(s)). Next: see tutorials/sim2real-blogs/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
