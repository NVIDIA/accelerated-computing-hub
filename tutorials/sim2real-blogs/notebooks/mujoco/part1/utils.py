# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
#
# Shared utilities for this series:
# - Menagerie asset download (verified sparse clone of one robot folder per pinned revision)
# - macOS mjpython relaunch for mujoco.viewer

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from robots import RobotSpec, get_robot

_EXECUTABLE_PATH_RE = re.compile(r"@executable_path/(.+) \(offset \d+\)\Z")


def _cache_base() -> Path:
    env = os.environ.get("MUJOCO_MENAGERIE_CACHE", "").strip()
    return Path(env).expanduser() if env else Path.home() / ".cache"


def default_cache_dir(spec: RobotSpec | None = None) -> Path:
    """Versioned automatic cache for *spec*: ``<base>/<cache_dirname>/<full ref>``.

    ``<base>`` is ``$MUJOCO_MENAGERIE_CACHE`` when set, otherwise ``~/.cache``.
    Each pinned revision gets its own folder, so an older cache (another
    revision or another layout) never blocks a first run. The
    accelerated-computing-hub copy of these lessons uses the same layout, so a
    valid cache downloaded by that copy is reused as is.
    """
    spec = spec or get_robot()
    return _cache_base() / spec.cache_dirname / spec.menagerie_ref


def _env_menagerie_root(spec: RobotSpec) -> Path | None:
    """Use MUJOCO_MENAGERIE_PATH only when this robot folder is actually present."""
    for env_name in ("MUJOCO_MENAGERIE_PATH", "NEWTON_MENAGERIE_PATH"):
        env_root = os.environ.get(env_name, "").strip()
        if not env_root:
            continue
        root = Path(env_root).expanduser()
        if (root / spec.folder).is_dir():
            return root
        print(
            f"Note: {env_name}={env_root} has no {spec.folder}/ folder; ignoring it and using "
            f"the automatic download (nothing to fix; `unset {env_name}` hides this note).",
            file=sys.stderr,
        )
    return None


def _git(*args: str, cwd: Path | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ["git", "--no-optional-locks"]
    if cwd is not None:
        cmd += ["-C", str(cwd)]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", LC_ALL="C")
    return subprocess.run([*cmd, *args], capture_output=capture, text=True, check=False, env=env)


def cache_problem(cache_root: Path, folder: str, expected_ref: str) -> str | None:
    """Why *cache_root* is not a clean checkout of *expected_ref*, or ``None`` if it is.

    Read-only: never changes the checkout. Only the selected robot subtree is
    inspected; unrelated assets and untracked output files are ignored. Git
    compares every tracked robot file against the pinned revision.
    """
    cache_root = Path(cache_root)
    if not (cache_root / ".git").exists():
        return "not a Git checkout"
    if not (cache_root / folder).is_dir():
        return f"{folder}/ is missing"
    try:
        head = _git("rev-parse", "--verify", "HEAD", cwd=cache_root)
        if head.returncode:
            return "cannot read the checkout revision"
        current = head.stdout.strip()
        if current != expected_ref:
            return f"checkout is at {current[:12]}, not the pinned {expected_ref[:12]}"
        status = _git(
            "status", "--porcelain=v1", "--untracked-files=no", "--", folder, cwd=cache_root
        )
        if status.returncode:
            return "cannot verify the robot files"
        if status.stdout.strip():
            return "tracked robot files are modified or missing"
    except OSError as exc:
        return f"cannot run Git ({exc})"
    return None


def validate_pinned_asset_cache(
    cache_root: Path, folder: str, expected_ref: str, *, cache_env: str
) -> None:
    """Raise if *cache_root* is not a clean checkout of *expected_ref* (read-only check).

    Explicit Menagerie path overrides intentionally bypass this check.
    """
    reason = cache_problem(cache_root, folder, expected_ref)
    if reason is not None:
        raise RuntimeError(
            f"Unverified asset cache {Path(cache_root) / folder}: expected {expected_ref} "
            f"({reason}). No cache files were changed. Set {cache_env} to a new empty "
            "directory and retry."
        )


class AssetDownloadError(RuntimeError):
    """A pinned asset could not be fetched; the message states the exact fix."""


def _download_failure(spec: RobotSpec, target: Path, detail: str) -> AssetDownloadError:
    return AssetDownloadError(
        f"Could not download the {spec.display_name} model ({spec.folder} @ "
        f"{spec.menagerie_ref[:12]}) from {spec.menagerie_url}: {detail}\n"
        "Fix: make sure this machine can reach github.com (VPN, proxy or firewall), then rerun "
        "the same command. The download happens once and is a sparse checkout of one robot folder.\n"
        "Offline machine: run the same command once on a connected machine, then copy the folder "
        f"{target} to the same path on this machine."
    )


def download_robot_sparse(spec: RobotSpec, cache_root: Path | None = None) -> Path:
    """Return ``<cache_root>/<spec.folder>`` at the pinned revision, downloading if needed.

    A valid existing checkout is reused without network access. Anything else
    already at *cache_root* (another revision, a partial download, another
    layout) is moved aside, never deleted. A new download goes into a temporary
    sibling folder and is renamed into place only after verification, so an
    interrupted run never leaves a half-written cache behind.
    """
    cache_root = Path(cache_root) if cache_root is not None else default_cache_dir(spec)
    robot_path = cache_root / spec.folder
    if cache_problem(cache_root, spec.folder, spec.menagerie_ref) is None:
        return robot_path

    if shutil.which("git") is None:
        raise _download_failure(
            spec, cache_root,
            "Git is not installed (Ubuntu: `sudo apt install git`; conda: `conda install git`).",
        )

    parent = cache_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{cache_root.name[:12]}-download-", dir=parent))
    print(
        f"Downloading the {spec.display_name} model ({spec.folder} @ {spec.menagerie_ref[:12]}) "
        f"from MuJoCo Menagerie, one time only, into {cache_root} ...",
        flush=True,
    )
    try:
        steps = (
            (False, "init", "-q", str(tmp)),
            (False, "-C", str(tmp), "remote", "add", "origin", spec.menagerie_url),
            (False, "-C", str(tmp), "sparse-checkout", "set", "--cone", spec.folder),
            # Network steps print Git's own progress (blobs arrive during checkout).
            (True, "-C", str(tmp), "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=60",
             "fetch", "--depth", "1", "--filter=blob:none", "origin", spec.menagerie_ref),
            (True, "-C", str(tmp), "-c", "advice.detachedHead=false", "checkout", "--detach",
             "FETCH_HEAD"),
        )
        for show, *args in steps:
            result = _git(*args, capture=not show)
            if result.returncode:
                err = (result.stderr or "").strip().splitlines()
                verb = next(a for a in args if a in ("init", "remote", "sparse-checkout", "fetch", "checkout"))
                raise _download_failure(
                    spec, cache_root,
                    f"`git {verb}` failed" + (f": {err[-1]}" if err else " (see the Git message above)"),
                )
        reason = cache_problem(tmp, spec.folder, spec.menagerie_ref)
        if reason is not None:
            raise _download_failure(spec, cache_root, f"the downloaded checkout is invalid ({reason})")

        if cache_root.exists() or cache_root.is_symlink():
            if cache_problem(cache_root, spec.folder, spec.menagerie_ref) is None:
                return robot_path  # another process finished first
            aside = cache_root.with_name(
                f"{cache_root.name}.unverified-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            cache_root.rename(aside)
            print(f"Moved the previous, unverified cache aside to {aside}", file=sys.stderr)
        tmp.rename(cache_root)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"Ready: {robot_path}", flush=True)
    return robot_path


def _legacy_cache_roots(spec: RobotSpec) -> list[Path]:
    """Older unversioned automatic layouts that may already hold the pinned checkout."""
    roots: list[Path] = []
    for root in (_cache_base() / spec.cache_dirname, Path.home() / ".cache" / spec.cache_dirname):
        if root not in roots:
            roots.append(root)
    return roots


def find_cached_robot(spec: RobotSpec) -> Path | None:
    """The verified cached robot folder, or ``None`` if a download is still needed."""
    for root in (default_cache_dir(spec), *_legacy_cache_roots(spec)):
        if cache_problem(root, spec.folder, spec.menagerie_ref) is None:
            return root / spec.folder
    return None


def resolve_menagerie_robot_path(
    spec: RobotSpec | None = None,
    menagerie_root: Path | None = None,
    *,
    explicit: bool = False,
) -> Path:
    """Return the robot folder, auto-downloading only that folder if needed.

    Order: ``--menagerie-path`` or ``MUJOCO_MENAGERIE_PATH`` (only when the robot
    folder exists there), a verified automatic cache (versioned layout, then the
    older unversioned layout), then a fresh download into the versioned cache.
    """
    spec = spec or get_robot()
    if menagerie_root is None:
        menagerie_root = _env_menagerie_root(spec)

    if menagerie_root is not None:
        robot_path = Path(menagerie_root).expanduser() / spec.folder
        if robot_path.is_dir():
            return robot_path
        if explicit:
            raise FileNotFoundError(
                f"--menagerie-path {menagerie_root} has no {spec.folder}/ folder. "
                "Omit --menagerie-path to download the pinned model automatically."
            )
        print(
            f"Note: {menagerie_root} has no {spec.folder}/ folder; using the automatic download.",
            file=sys.stderr,
        )

    return find_cached_robot(spec) or download_robot_sparse(spec, default_cache_dir(spec))


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


def doctor_command() -> str:
    """Command line for the repository's setup checker."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "doctor.py").is_file():
            return f"python {parent / 'doctor.py'}"
    return "python doctor.py   # from the repository root"


def print_asset_setup_help(script_name: str) -> None:
    """Follow-up for asset errors: the error above already states the exact fix."""
    print(
        "\nRobot models download automatically on first use; no environment variable is needed.\n"
        "To check the whole setup and print the fix for each problem, run:\n"
        f"  {doctor_command()}\n",
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
