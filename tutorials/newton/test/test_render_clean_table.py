"""Offline exporter protocol fixtures, NOT simulated physics outcomes."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from clean_table_report_fixture import make_gripper_report

TOOL = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "tools" / "render_clean_table.py"
spec = importlib.util.spec_from_file_location("blog3_clean_renderer", TOOL)
assert spec is not None and spec.loader is not None
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def write_record_fixture(path, *, offset=0.0):
    """Minimal draw-able arrays; never used as article evidence."""
    report = make_gripper_report()
    body = np.asarray([[[offset, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]] * 6)
    cloth = np.asarray([[[0.2, 0.0, 0.15], [0.22, 0.0, 0.15], [0.2, 0.02, 0.15]]] * 6)
    np.savez_compressed(
        path, time=[0.0, 1.5, 4.5, 7.5, 10.5, 13.0], body_q=body, particle_q=cloth,
        phase=["settle", "carry", "carry", "carry", "carry", "done"],
        active_object=["", "red_cube", "blue_cube", "shirt", "cable", ""],
        shape_type=[7], shape_body=[0], shape_scale=[[0.02, 0.02, 0.02]],
        shape_transform=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], shape_color=[[0.3, 0.7, 0.3]],
        joint_parent=[-1], joint_child=[0], tri_indices=[[0, 1, 2]],
        bin_lower=report["bin_lower"], bin_upper=report["bin_upper"],
    )
    report["record_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


class RecordingBindingTests(unittest.TestCase):
    def test_matching_pair_exports_only_a_temporary_fixture_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, report_path, output = root / "fixture.npz", root / "fixture.json", root / "figure"
            report = write_record_fixture(record)
            report_path.write_text(json.dumps(report))
            # Binding is by content, not basename: moving a valid pair is safe.
            renamed = record.rename(root / "renamed.npz")
            self.run_renderer(renamed, report_path, output)
            image = output / "clean-table-so101.png"
            self.assertTrue(image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_missing_malformed_hashes_and_failed_reports_do_not_create_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, report_path, output = root / "fixture.npz", root / "fixture.json", root / "figure"
            valid = write_record_fixture(record)
            variants = [{key: value for key, value in valid.items() if key != "record_sha256"},
                        {**valid, "success": False}]
            variants.extend({**valid, "record_sha256": value} for value in (
                None, True, "", "g" * 64, valid["record_sha256"] + " ", valid["record_sha256"].upper(),
            ))
            for report in variants:
                with self.subTest(report=report):
                    report_path.write_text(json.dumps(report))
                    with mock.patch.object(renderer.plt, "figure") as figure:
                        with self.assertRaisesRegex(SystemExit, "SHA-256|did not pass"):
                            self.run_renderer(record, report_path, output)
                        figure.assert_not_called()
                    self.assertFalse(output.exists())

    def test_modifying_recorded_bytes_invalidates_a_previously_matching_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, report_path, output = root / "fixture.npz", root / "fixture.json", root / "figure"
            report_path.write_text(json.dumps(write_record_fixture(record)))
            # ZIP trailing bytes can still decode; the byte binding must reject them.
            with record.open("ab") as stream:
                stream.write(b"altered after report")
            with mock.patch.object(renderer.plt, "figure") as figure:
                with self.assertRaisesRegex(SystemExit, "SHA-256"):
                    self.run_renderer(record, report_path, output)
                figure.assert_not_called()
            self.assertFalse(output.exists())

    def test_sweep_or_missing_grasp_evidence_cannot_be_exported_as_pick_and_place(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, report_path, output = root / "fixture.npz", root / "fixture.json", root / "figure"
            valid = write_record_fixture(record)
            missing_grasp = json.loads(json.dumps(valid))
            del missing_grasp["objects"]["cable"]["carry_time"]
            for report in ({**valid, "task": "physical_sweep_into_bin"}, missing_grasp):
                report_path.write_text(json.dumps(report))
                with mock.patch.object(renderer.plt, "figure") as figure:
                    with self.assertRaisesRegex(SystemExit, "unverified gripper task"):
                        self.run_renderer(record, report_path, output)
                    figure.assert_not_called()
                self.assertFalse(output.exists())

    def test_story_frames_follow_each_recorded_object_carry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            write_record_fixture(path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual([frame for frame, _ in renderer.story_frames(data)], list(range(6)))
                arrays = dict(data)
            reversed_materials = dict(arrays)
            reversed_materials["active_object"] = arrays["active_object"].copy()
            reversed_materials["active_object"][[3, 4]] = ["cable", "shirt"]
            selected = renderer.story_frames(reversed_materials)
            self.assertEqual([frame for frame, _ in selected], list(range(6)))
            self.assertTrue(selected[3][1].startswith("Cable:"))
            arrays["active_object"][4] = "shirt"
            with self.assertRaisesRegex(ValueError, "no carry frames for cable"):
                renderer.story_frames(arrays)
            arrays["phase"] = arrays["phase"][:-1]
            with self.assertRaisesRegex(ValueError, "align"):
                renderer.story_frames(arrays)

    def run_renderer(self, record, report, output):
        with mock.patch.object(sys, "argv", [str(TOOL), "--record", str(record), "--report", str(report),
                                             "--output-dir", str(output)]), contextlib.redirect_stdout(io.StringIO()):
            renderer.main()

    def test_same_duration_different_record_is_rejected_before_creating_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, other = root / "failed-fixture.npz", root / "other-fixture.npz"
            failed_report = write_record_fixture(record)
            failed_report["success"] = False
            (root / "failed-fixture.json").write_text(json.dumps(failed_report))
            report = write_record_fixture(other, offset=0.1)
            report_path = root / "other-fixture.json"
            report_path.write_text(json.dumps(report))
            self.assertNotEqual(failed_report["record_sha256"], report["record_sha256"])
            self.assertEqual(failed_report["simulation_seconds"], report["simulation_seconds"])
            output = root / "must-not-exist"
            with mock.patch.object(renderer.plt, "figure", wraps=renderer.plt.figure) as figure:
                with self.assertRaisesRegex(SystemExit, "SHA-256"):
                    self.run_renderer(record, report_path, output)
                figure.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
