#!/usr/bin/env python3
"""Download the exact runtime assets for the pinned ALOHA replay reference.

This helper does not confer rights to third-party assets. Read the upstream
attributions and the accompanying publication recommendation before reuse.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.request

MANIFEST_SHA256 = "87790d12695c9639d09f74a4562e2fa66271abb522e2bdbd48acc1cc52759831"
REPOS = {"google-deepmind/mujoco_warp", "google-deepmind/mujoco_menagerie"}


def checked_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or any(x in ("", ".", "..") for x in path.parts) or "\\" in value:
        raise ValueError(f"Unsafe manifest path: {value}")
    return path


def verified(data, item):
    return len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true", help="Check an existing cache without network access")
    args = parser.parse_args()
    raw = Path(__file__).with_name("assets-manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Asset manifest hash mismatch")
    manifest = json.loads(raw)
    root = args.output.expanduser().resolve()
    if not args.verify_only:
        root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for item in manifest["files"]:
        rel = checked_path(item["path"])
        upstream = checked_path(item["upstream_path"])
        if item["upstream_repo"] not in REPOS or not re.fullmatch(r"[0-9a-f]{40}", item["commit"]):
            raise ValueError("Unrecognized pinned upstream")
        target = root / rel
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Cache path escapes output root: {target}")
        if target.exists():
            if not verified(target.read_bytes(), item):
                raise ValueError(f"Existing file differs from pinned source; preserved: {target}")
            continue
        if args.verify_only:
            raise FileNotFoundError(target)
        url = f'https://raw.githubusercontent.com/{item["upstream_repo"]}/{item["commit"]}/{upstream}'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(item["bytes"] + 1)
        if not verified(data, item):
            raise ValueError(f"Downloaded asset fails byte count or SHA256 check: {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".asset-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            # Exclusive creation preserves pre-existing content even if another
            # process populated this cache after the check above.
            os.link(name, target)
        finally:
            os.unlink(name)
        downloaded += 1
    print(json.dumps({"verified_files": len(manifest["files"]), "downloaded_files": downloaded,
                      "bytes": sum(x["bytes"] for x in manifest["files"]),
                      "manifest_sha256": MANIFEST_SHA256, "output": str(root)}))


if __name__ == "__main__":
    main()
