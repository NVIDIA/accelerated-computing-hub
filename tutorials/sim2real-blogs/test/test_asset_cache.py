"""Automatic asset caches: pinned, verified, never a dead end; custom paths are deliberate."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PART1 = Path(__file__).resolve().parents[1] / "notebooks" / "mujoco" / "part1"
sys.path.insert(0, str(PART1))

from robots import get_robot
from utils import (
    AssetDownloadError,
    cache_problem,
    default_cache_dir,
    download_robot_sparse,
    print_asset_setup_help,
    resolve_menagerie_robot_path,
    validate_pinned_asset_cache,
)


class AssetCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.root = self.tmp / "assets"
        self.root.mkdir()
        self.folder = "robotstudio_so101"
        self.robot = self.root / self.folder
        self.robot.mkdir()
        self.xml = self.robot / "so101.xml"
        self.xml.write_text("<mujoco/>\n")
        (self.robot / "mesh.stl").write_bytes(b"original mesh")
        (self.root / "other-robot.txt").write_text("unrelated asset\n")
        self.git("init", "-q")
        self.git("config", "user.name", "Asset cache test")
        self.git("config", "user.email", "asset-cache-test@example.invalid")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "Pinned fixture")
        self.pin = self.git("rev-parse", "HEAD")
        # A local origin keeps every download test offline and deterministic.
        self.spec = replace(get_robot("so101"), menagerie_ref=self.pin,
                            menagerie_url=self.root.as_uri())
        self.base = self.tmp / "cache-base"
        env = {k: v for k, v in os.environ.items()
               if k not in ("MUJOCO_MENAGERIE_PATH", "NEWTON_MENAGERIE_PATH")}
        env["MUJOCO_MENAGERIE_CACHE"] = str(self.base)
        self.enterContext(mock.patch.dict("os.environ", env, clear=True))

    def git(self, *args, cwd=None):
        return subprocess.run(
            ["git", "-C", str(cwd or self.root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def validate(self):
        validate_pinned_asset_cache(
            self.root, self.folder, self.pin, cache_env="MUJOCO_MENAGERIE_CACHE"
        )

    def resolve_quietly(self, spec=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            path = resolve_menagerie_robot_path(spec or self.spec)
        return path, out.getvalue(), err.getvalue()

    # -- verification of an existing checkout (read-only) ---------------------------------

    def test_clean_cache_is_reused_without_network_or_checkout_changes(self):
        # A scratch output and a different robot's changes do not affect this asset.
        (self.robot / "render.log").write_text("scratch output")
        (self.root / "other-robot.txt").write_text("local unrelated edit")
        with mock.patch("utils.subprocess.run", wraps=subprocess.run) as run:
            self.assertEqual(download_robot_sparse(self.spec, self.root), self.robot)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.pin)
        for call in run.call_args_list:
            self.assertFalse({"fetch", "clone", "checkout", "reset"} & set(call.args[0]))

    def test_wrong_revision_reports_expected_current_and_path_without_repair(self):
        expected = "0" * 40
        with self.assertRaises(RuntimeError) as caught:
            validate_pinned_asset_cache(self.root, self.folder, expected,
                                        cache_env="MUJOCO_MENAGERIE_CACHE")
        message = str(caught.exception)
        for value in (str(self.robot), expected, self.pin[:12], "MUJOCO_MENAGERIE_CACHE"):
            self.assertIn(value, message)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.pin)
        self.assertEqual(self.xml.read_text(), "<mujoco/>\n")

    def test_modified_tracked_asset_is_rejected_and_preserved(self):
        self.xml.write_text("<mujoco model='local edit'/>\n")
        with self.assertRaisesRegex(RuntimeError, "tracked robot files are modified or missing"):
            self.validate()
        self.assertEqual(self.xml.read_text(), "<mujoco model='local edit'/>\n")
        self.assertEqual(self.git("rev-parse", "HEAD"), self.pin)

    def test_staged_asset_change_is_rejected(self):
        self.xml.write_text("<mujoco model='staged edit'/>\n")
        self.git("add", self.folder)
        with self.assertRaisesRegex(RuntimeError, "tracked robot files are modified or missing"):
            self.validate()

    def test_missing_tracked_mesh_is_rejected(self):
        (self.robot / "mesh.stl").unlink()
        with self.assertRaisesRegex(RuntimeError, "tracked robot files are modified or missing"):
            self.validate()

    # -- first run never dead-ends --------------------------------------------------------

    def test_default_cache_is_versioned_by_full_pin(self):
        self.assertEqual(default_cache_dir(self.spec), self.base / "mujoco_menagerie" / self.pin)

    def test_tester_report_old_nonempty_cache_and_missing_override_still_runs(self):
        # 13 Oct report: ~/.cache/mujoco_menagerie held another layout's files and
        # MUJOCO_MENAGERIE_PATH pointed to a folder that does not exist.
        old = self.base / "mujoco_menagerie"
        (old / ("f" * 40)).mkdir(parents=True)
        (old / ("f" * 40) / "leftover.txt").write_text("older layout")
        (old / "notes.txt").write_text("user file")
        os.environ["MUJOCO_MENAGERIE_PATH"] = str(self.tmp / "mujoco_menagerie")
        path, out, err = self.resolve_quietly()
        self.assertEqual(path, old / self.pin / self.folder)
        self.assertEqual((path / "so101.xml").read_text(), "<mujoco/>\n")
        self.assertIsNone(cache_problem(old / self.pin, self.folder, self.pin))
        self.assertEqual((old / "notes.txt").read_text(), "user file")
        self.assertEqual((old / ("f" * 40) / "leftover.txt").read_text(), "older layout")
        self.assertIn("Downloading", out)
        self.assertIn("one time only", out)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertIn("ignoring it and using the automatic download", err)
        self.assertNotIn("export", err)
        self.assertFalse([p for p in old.iterdir() if p.name.startswith(".")], "temp dir left")

    def test_second_run_reuses_the_download_without_network(self):
        self.resolve_quietly()
        with mock.patch("utils.subprocess.run", wraps=subprocess.run) as run:
            path, out, _ = self.resolve_quietly()
        self.assertEqual(path, self.base / "mujoco_menagerie" / self.pin / self.folder)
        self.assertNotIn("Downloading", out)
        for call in run.call_args_list:
            self.assertFalse({"fetch", "clone", "init"} & set(call.args[0]))

    def test_cache_from_the_hub_layout_is_reused(self):
        # accelerated-computing-hub (8 Oct) cloned into <cache>/<dirname>/<full ref>.
        hub = self.base / "mujoco_menagerie" / self.pin
        hub.parent.mkdir(parents=True)
        subprocess.run(["git", "clone", "-q", str(self.root), str(hub)], check=True)
        path, out, _ = self.resolve_quietly()
        self.assertEqual(path, hub / self.folder)
        self.assertNotIn("Downloading", out)

    def test_valid_cache_in_the_older_unversioned_layout_is_reused(self):
        legacy = self.base / "mujoco_menagerie"
        subprocess.run(["git", "clone", "-q", str(self.root), str(legacy)], check=True)
        path, out, _ = self.resolve_quietly()
        self.assertEqual(path, legacy / self.folder)
        self.assertNotIn("Downloading", out)

    def test_unverified_versioned_cache_is_moved_aside_and_replaced(self):
        target = self.base / "mujoco_menagerie" / self.pin
        (target / self.folder).mkdir(parents=True)
        (target / self.folder / "so101.xml").write_text("user asset")
        path, _, err = self.resolve_quietly()
        self.assertEqual(path, target / self.folder)
        self.assertIsNone(cache_problem(target, self.folder, self.pin))
        aside = [p for p in target.parent.iterdir() if ".unverified-" in p.name]
        self.assertEqual(len(aside), 1)
        self.assertEqual((aside[0] / self.folder / "so101.xml").read_text(), "user asset")
        self.assertIn("Moved the previous, unverified cache aside", err)

    def test_offline_download_fails_once_with_the_exact_fix_and_leaves_nothing(self):
        offline = replace(self.spec, menagerie_url=(self.tmp / "no-network.git").as_uri())
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(AssetDownloadError) as caught:
                resolve_menagerie_robot_path(offline)
        message = str(caught.exception)
        self.assertIn("Fix:", message)
        self.assertIn("github.com", message)
        self.assertIn(str(default_cache_dir(offline)), message)
        self.assertNotIn("MUJOCO_MENAGERIE_CACHE to another folder", message)
        parent = self.base / "mujoco_menagerie"
        self.assertEqual(list(parent.iterdir()) if parent.exists() else [], [])
        help_text = io.StringIO()
        with redirect_stderr(help_text):
            print_asset_setup_help("so101_pick_place.py")
        self.assertIn("doctor.py", help_text.getvalue())
        self.assertNotIn("export MUJOCO_MENAGERIE_PATH", help_text.getvalue())

    def test_explicit_path_and_both_environment_overrides_remain_authoritative(self):
        # Deliberate user assets may be edited or unversioned, unlike auto-caches.
        root = self.tmp / "custom"
        (root / self.folder).mkdir(parents=True)
        with mock.patch("utils.download_robot_sparse") as download:
            self.assertEqual(
                resolve_menagerie_robot_path(self.spec, root, explicit=True), root / self.folder
            )
            for name in ("MUJOCO_MENAGERIE_PATH", "NEWTON_MENAGERIE_PATH"):
                with self.subTest(name=name), mock.patch.dict("os.environ", {name: str(root)}, clear=True):
                    self.assertEqual(resolve_menagerie_robot_path(self.spec), root / self.folder)
            download.assert_not_called()

    def test_missing_explicit_menagerie_path_is_a_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Omit --menagerie-path"):
            resolve_menagerie_robot_path(self.spec, self.tmp / "nowhere", explicit=True)

    def test_fresh_sparse_download_checks_out_requested_pin(self):
        # The local origin also has a newer, different revision.
        self.xml.write_text("<mujoco model='newer revision'/>\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "Unrequested newer revision")
        cache = self.tmp / "new-cache"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            path = download_robot_sparse(self.spec, cache)
        self.assertEqual((path / "so101.xml").read_text(), "<mujoco/>\n")
        self.assertFalse((cache / "other-robot.txt").is_dir())
        validate_pinned_asset_cache(cache, self.folder, self.pin, cache_env="MUJOCO_MENAGERIE_CACHE")


if __name__ == "__main__":
    unittest.main()
