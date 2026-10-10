"""Blog 2 asset selection/download contracts, with all Git commands mocked.

Existing robot folders are reused by the measured source; these tests do not
misrepresent reuse as a Git revision or tracked-file attestation.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

PART1 = Path(__file__).resolve().parents[1] / "notebooks/mujoco/part1"
sys.path.insert(0, str(PART1))
try:
    from robots import get_robot
    spec = importlib.util.spec_from_file_location("hub_blog2_assets", PART1 / "utils.py")
    assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assets)
finally:
    sys.path.remove(str(PART1))


class AssetCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.spec = get_robot("so101")

    def test_default_cache_selection_is_robot_specific(self):
        with mock.patch.dict("os.environ", {"MUJOCO_MENAGERIE_CACHE": str(self.root)}, clear=True):
            for robot in ("so101", "rebot"):
                spec = get_robot(robot)
                self.assertEqual(assets.default_cache_dir(spec), self.root / spec.cache_dirname)

    def test_existing_folder_is_preserved_without_claiming_attestation(self):
        folder = self.root / self.spec.folder
        folder.mkdir()
        marker = folder / "custom.txt"
        marker.write_text("existing user material")
        with mock.patch.object(assets.subprocess, "run") as run:
            self.assertEqual(assets.download_robot_sparse(self.spec, self.root), folder)
            run.assert_not_called()
        self.assertEqual(marker.read_text(), "existing user material")

    def test_fresh_download_selects_exact_robot_and_pinned_ref(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if "checkout" in command:
                (self.root / self.spec.folder).mkdir()
            return SimpleNamespace(returncode=0)
        with mock.patch.object(assets.subprocess, "run", side_effect=run):
            self.assertEqual(assets.download_robot_sparse(self.spec, self.root), self.root / self.spec.folder)
        self.assertIn(["git", "-C", str(self.root), "sparse-checkout", "set", self.spec.folder], calls)
        self.assertIn(["git", "-C", str(self.root), "fetch", "--depth", "1", "origin", self.spec.menagerie_ref], calls)
        self.assertIn(["git", "-C", str(self.root), "checkout", "FETCH_HEAD"], calls)
        self.assertEqual(calls[0][-2:], [self.spec.menagerie_url, str(self.root)])

    def test_nonempty_target_is_refused_without_deleting_files(self):
        marker = self.root / "unrelated.txt"
        marker.write_text("keep")
        with mock.patch.object(assets.subprocess, "run") as run, self.assertRaisesRegex(RuntimeError, "non-empty"):
            assets.download_robot_sparse(self.spec, self.root)
        run.assert_not_called()
        self.assertEqual(marker.read_text(), "keep")

    def test_missing_downloaded_robot_is_not_success(self):
        with mock.patch.object(assets.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                self.assertRaisesRegex(RuntimeError, "missing"):
            assets.download_robot_sparse(self.spec, self.root)

    def test_git_failure_propagates(self):
        failure = subprocess.CalledProcessError(1, ["git", "clone"])
        with mock.patch.object(assets.subprocess, "run", side_effect=failure), \
                self.assertRaises(subprocess.CalledProcessError):
            assets.download_robot_sparse(self.spec, self.root)

    def test_explicit_and_environment_paths_remain_authoritative(self):
        folder = self.root / self.spec.folder
        folder.mkdir()
        with mock.patch.object(assets, "download_robot_sparse") as download:
            self.assertEqual(assets.resolve_menagerie_robot_path(self.spec, self.root, explicit=True), folder)
            for name in ("MUJOCO_MENAGERIE_PATH", "NEWTON_MENAGERIE_PATH"):
                with mock.patch.dict("os.environ", {name: str(self.root)}, clear=True):
                    self.assertEqual(assets.resolve_menagerie_robot_path(self.spec), folder)
            download.assert_not_called()

    def test_missing_explicit_path_fails_before_download(self):
        with mock.patch.object(assets, "download_robot_sparse") as download, \
                self.assertRaises(FileNotFoundError):
            assets.resolve_menagerie_robot_path(self.spec, self.root, explicit=True)
        download.assert_not_called()
