# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# Shared utilities for this series:
# - Menagerie asset download (sparse clone of one robot folder)
# - macOS mjpython relaunch for mujoco.viewer

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from robots import RobotSpec, get_robot

_EXECUTABLE_PATH_RE = re.compile(r"@executable_path/(.+) \(offset \d+\)\Z")


def default_cache_dir(spec: RobotSpec | None = None) -> Path:
    spec = spec or get_robot()
    if "MUJOCO_MENAGERIE_CACHE" in os.environ:
        return Path(os.environ["MUJOCO_MENAGERIE_CACHE"]) / spec.cache_dirname
    return Path.home() / ".cache" / spec.cache_dirname


def _env_menagerie_root(spec: RobotSpec) -> Path | None:
    """Use MUJOCO_MENAGERIE_PATH only when this robot folder is actually present."""
    for env_name in ("MUJOCO_MENAGERIE_PATH", "NEWTON_MENAGERIE_PATH"):
        env_root = os.environ.get(env_name, "").strip()
        if not env_root:
            continue
        root = Path(env_root).expanduser()
        robot_path = root / spec.folder
        if robot_path.exists():
            return root
        print(
            f"Ignoring {env_name}={env_root!r} ({robot_path} not found). "
            f"Unset it with: unset {env_name}",
            file=sys.stderr,
        )
    return None


def download_robot_sparse(spec: RobotSpec, cache_root: Path) -> Path:
    """Sparse-clone only ``spec.folder`` into cache_root."""
    robot_path = cache_root / spec.folder
    if robot_path.exists():
        return robot_path

    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {spec.folder} from {spec.menagerie_url} (ref {spec.menagerie_ref[:12]})...")
    print(f"Cache: {cache_root}")

    if (cache_root / ".git").is_dir():
        subprocess.run(
            ["git", "-C", str(cache_root), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cache_root), "sparse-checkout", "set", spec.folder],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cache_root), "fetch", "--depth", "1", "origin", spec.menagerie_ref],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cache_root), "checkout", "FETCH_HEAD"],
            check=True,
        )
    else:
        if any(cache_root.iterdir()):
            raise RuntimeError(
                f"Cannot download into non-empty {cache_root}. "
                f"Remove it or set MUJOCO_MENAGERIE_CACHE to another folder."
            )
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                spec.menagerie_url,
                str(cache_root),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cache_root), "sparse-checkout", "set", spec.folder],
            check=True,
        )
        if spec.menagerie_ref not in ("main", "master", "HEAD"):
            fetch = subprocess.run(
                ["git", "-C", str(cache_root), "fetch", "--depth", "1", "origin", spec.menagerie_ref],
                check=False,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(cache_root),
                    "checkout",
                    "FETCH_HEAD" if fetch.returncode == 0 else spec.menagerie_ref,
                ],
                check=True,
            )

    if not robot_path.exists():
        raise RuntimeError(f"Download finished but {robot_path} is missing.")
    print(f"Ready: {robot_path}")
    return robot_path


def resolve_menagerie_robot_path(
    spec: RobotSpec | None = None,
    menagerie_root: Path | None = None,
    *,
    explicit: bool = False,
) -> Path:
    """Return the robot folder, auto-downloading only that folder if needed."""
    spec = spec or get_robot()
    if menagerie_root is None:
        menagerie_root = _env_menagerie_root(spec)

    if menagerie_root is not None:
        robot_path = menagerie_root / spec.folder
        if robot_path.exists():
            return robot_path
        if explicit:
            raise FileNotFoundError(
                f"Expected {robot_path} but it does not exist. "
                f"Clone the menagerie or omit --menagerie-path to auto-download."
            )
        print(
            f"--menagerie-path {menagerie_root} has no {spec.folder}/; auto-downloading instead.",
            file=sys.stderr,
        )

    return download_robot_sparse(spec, default_cache_dir(spec))


def resolve_scene_xml(
    spec: RobotSpec | None = None,
    menagerie_root: Path | None = None,
    *,
    scene: str = "scene",
    explicit: bool = False,
) -> Path:
    """Resolve menagerie assets and return the vendor scene XML path."""
    spec = spec or get_robot()
    robot_path = resolve_menagerie_robot_path(spec, menagerie_root, explicit=explicit)
    scene_name = "scene.xml" if scene == "scene" else "scene_box.xml"
    xml_path = robot_path / scene_name
    if not xml_path.exists():
        raise FileNotFoundError(f"Scene file not found: {xml_path}")
    return xml_path


def standalone_viewer_command(xml_path: Path) -> list[str]:
    """Command for the managed standalone viewer (plain python works on macOS)."""
    return [sys.executable, "-m", "mujoco.viewer", "--mjcf", str(xml_path)]


def print_asset_setup_help(script_name: str) -> None:
    print(
        "\nAuto-download (default — only the selected robot folder):\n"
        "  unset MUJOCO_MENAGERIE_PATH   # if set to a placeholder path\n"
        f"  python {script_name} --robot so101\n"
        f"  python {script_name} --robot rebot\n"
        "\nOr point to a real local clone that contains the robot folder:\n"
        "  export MUJOCO_MENAGERIE_PATH=$HOME/mujoco_menagerie\n",
        file=sys.stderr,
    )


def running_under_mjpython() -> bool:
    if os.environ.get("MJPYTHON_BIN"):
        return True
    return "mjpython" in Path(sys.executable).name.lower()


def find_mjpython_bin() -> Path | None:
    """Path to the native mjpython binary shipped with the mujoco package."""
    spec = importlib.util.find_spec("mujoco")
    if not spec or not spec.origin:
        return None
    candidate = Path(spec.origin).parent / "MuJoCo_(mjpython).app/Contents/MacOS/mjpython"
    return candidate if candidate.is_file() else None


def find_mjpython() -> str | None:
    """Human-facing launcher hint (auto-relaunch uses the native binary directly)."""
    if path := shutil.which("mjpython"):
        return path
    spec = importlib.util.find_spec("mujoco")
    if spec and spec.origin:
        wrapper = Path(spec.origin).parent / "mjpython" / "mjpython.py"
        if wrapper.is_file():
            return str(wrapper)
    return None


def _otool_dyld_fallback_paths(binary: str) -> list[str]:
    """Mirror mujoco's mjpython.py @executable_path dylib resolution."""
    libpython_dir = os.path.dirname(binary)
    dyld_fallback_paths: list[str] = []
    try:
        otool_out = subprocess.run(
            ["otool", "-l", binary],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return dyld_fallback_paths

    for line in otool_out.split("\n"):
        match = _EXECUTABLE_PATH_RE.search(line)
        if match is not None:
            new_path = os.path.dirname(os.path.join(libpython_dir, match.group(1)))
            if new_path not in dyld_fallback_paths:
                dyld_fallback_paths.insert(0, new_path)
    return dyld_fallback_paths


def _resolve_mjpython_libpython() -> tuple[str, list[str]]:
    """Resolve libpython for mjpython; fixes uv venv shims on macOS."""
    major, minor = sys.version_info.major, sys.version_info.minor
    base = Path(sys.base_prefix)
    lib_dir = base / "lib"
    dylib = lib_dir / f"libpython{major}.{minor}.dylib"

    if dylib.is_file():
        for name in (f"python{major}.{minor}", f"python{major}", "python"):
            candidate = base / "bin" / name
            if candidate.exists():
                libpython = str(candidate.resolve())
                fallback = _otool_dyld_fallback_paths(libpython)
                if str(lib_dir) not in fallback:
                    fallback.insert(0, str(lib_dir))
                return libpython, fallback
        return str(dylib), [str(lib_dir)]

    libpython = str(Path(sys.executable).resolve())
    return libpython, _otool_dyld_fallback_paths(libpython)


def wants_headless(argv: list[str]) -> bool:
    for i, arg in enumerate(argv):
        if arg == "--headless-steps" and i + 1 < len(argv):
            try:
                return int(argv[i + 1]) > 0
            except ValueError:
                return False
        if arg.startswith("--headless-steps="):
            try:
                return int(arg.split("=", 1)[1]) > 0
            except ValueError:
                return False
        # Newton examples use --viewer null / --num-frames for headless.
        if arg in ("--viewer",) and i + 1 < len(argv) and argv[i + 1] == "null":
            return True
        if arg == "--headless":
            return True
        # Throughput / scaling runs never open mujoco.viewer.
        if arg == "--benchmark":
            return True
    return False


def maybe_relaunch_with_mjpython() -> None:
    """macOS MuJoCo viewer requires mjpython; relaunch automatically."""
    if sys.platform != "darwin" or wants_headless(sys.argv[1:]) or running_under_mjpython():
        return
    mjpython_bin = find_mjpython_bin()
    if mjpython_bin is None:
        return

    # Bypass mujoco's mjpython.py wrapper: it dlopens sys.executable, which breaks
    # uv/virtualenv shims that don't ship libpython next to bin/python3.
    libpython_path, dyld_fallback = _resolve_mjpython_libpython()
    env = os.environ.copy()
    env["MJPYTHON_BIN"] = str(mjpython_bin)
    env["MJPYTHON_LIBPYTHON"] = libpython_path

    if dyld_fallback:
        if existing := env.get("DYLD_FALLBACK_LIBRARY_PATH", ""):
            if existing:
                dyld_fallback.extend(existing.split(":"))
        else:
            dyld_fallback.extend(("/usr/local/lib", "/usr/lib"))
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dyld_fallback)

    argv = [sys.executable, *(str(Path(a).resolve()) if i == 0 else a for i, a in enumerate(sys.argv))]
    print("macOS: relaunching under mjpython (required for mujoco.viewer)...")
    os.execve(str(mjpython_bin), argv, env)


def mjpython_viewer_error(script_name: str, exc: RuntimeError) -> RuntimeError:
    """Return a clearer viewer error for macOS mjpython requirement."""
    if "mjpython" not in str(exc).lower() or sys.platform != "darwin":
        return exc
    mjpython = find_mjpython()
    hint = f"  {mjpython} {script_name}\n" if mjpython else ""
    return RuntimeError(
        "MuJoCo passive viewer on macOS must run under mjpython.\n"
        f"  mjpython {script_name}\n"
        f"{hint}"
        f"  Or headless: python {script_name} --headless-steps 100"
    )
