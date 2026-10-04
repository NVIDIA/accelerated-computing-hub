"""Automatic asset caches must be pinned and clean; custom paths are deliberate."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PART3 = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"
sys.path.insert(0, str(PART3))

from robots import get_robot
from utils import download_robot_sparse, resolve_menagerie_robot_path, validate_pinned_asset_cache


class AssetCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "assets"
        self.root.mkdir()
        self.folder = "robotstudio_so101"
        self.robot = self.root / self.folder
        self.robot.mkdir()
        self.xml = self.robot / "so101.xml"
        self.xml.write_text("<mujoco/>\n")
        (self.robot / "mesh.stl").write_bytes(b"original mesh")
        (self.root / "other-robot.txt").write_text("unrelated asset\n")
        self.git("init")
        self.git("config", "user.name", "Asset cache test")
        self.git("config", "user.email", "asset-cache-test@example.invalid")
        self.git("add", ".")
        self.git("commit", "-m", "Pinned fixture")
        self.pin = self.git("rev-parse", "HEAD")
        self.spec = replace(get_robot("so101"), menagerie_ref=self.pin)

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def validate(self):
        validate_pinned_asset_cache(
            self.root, self.folder, self.pin, cache_env="MUJOCO_MENAGERIE_CACHE"
        )

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
            download_robot_sparse(replace(self.spec, menagerie_ref=expected), self.root)
        message = str(caught.exception)
        for value in (str(self.robot), expected, self.pin, "MUJOCO_MENAGERIE_CACHE"):
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

    def test_unversioned_automatic_cache_is_not_trusted(self):
        root = self.root.parent / "unversioned"
        robot = root / self.folder
        robot.mkdir(parents=True)
        (robot / "so101.xml").write_text("user asset")
        with self.assertRaisesRegex(RuntimeError, "Git metadata is missing"):
            download_robot_sparse(self.spec, root)
        self.assertEqual((robot / "so101.xml").read_text(), "user asset")

    def test_incomplete_nonempty_cache_is_not_checked_out_or_deleted(self):
        missing = replace(self.spec, folder="missing_robot")
        with self.assertRaisesRegex(RuntimeError, "incomplete asset cache"):
            download_robot_sparse(missing, self.root)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.pin)
        self.assertEqual(self.xml.read_text(), "<mujoco/>\n")

    def test_explicit_path_and_both_environment_overrides_remain_authoritative(self):
        # Deliberate user assets may be edited or unversioned, unlike auto-caches.
        root = self.root.parent / "custom"
        (root / self.folder).mkdir(parents=True)
        with mock.patch("utils.download_robot_sparse") as download:
            self.assertEqual(
                resolve_menagerie_robot_path(self.spec, root, explicit=True), root / self.folder
            )
            for name in ("MUJOCO_MENAGERIE_PATH", "NEWTON_MENAGERIE_PATH"):
                with self.subTest(name=name), mock.patch.dict("os.environ", {name: str(root)}, clear=True):
                    self.assertEqual(resolve_menagerie_robot_path(self.spec), root / self.folder)
            download.assert_not_called()

    def test_fresh_sparse_download_checks_out_requested_pin(self):
        # Local origin avoids the network and includes a newer, different revision.
        self.xml.write_text("<mujoco model='newer revision'/>\n")
        self.git("add", ".")
        self.git("commit", "-m", "Unrequested newer revision")
        spec = replace(self.spec, menagerie_url=self.root.as_uri())
        cache = self.root.parent / "new-cache"
        path = download_robot_sparse(spec, cache)
        self.assertEqual((path / "so101.xml").read_text(), "<mujoco/>\n")
        validate_pinned_asset_cache(cache, self.folder, self.pin, cache_env="MUJOCO_MENAGERIE_CACHE")


if __name__ == "__main__":
    unittest.main()
