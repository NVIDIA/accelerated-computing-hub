# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Workflow benchmark checks using real MuJoCo task episodes on CPU."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import mujoco
import numpy as np

PART3 = Path(__file__).resolve().parents[1] / "notebooks" / "newton" / "part3"
sys.path.insert(0, str(PART3))

import benchmark_workflows as workflow
import pick_place_common as task


class WorkflowBenchmarkTests(unittest.TestCase):
    def test_invalid_configuration_cannot_produce_a_comparison(self):
        for invalid in ({"scope": "replay"}, {"backend": "mjwarp"}, {"robot": "unknown"},
                        {"frames": 100}, {"substeps": 1}, {"worlds": 2}, {"cpu_threads": 2},
                        {"warmups": 0}, {"repeats": 0}, {"warmups": 1.5}, {"repeats": True},
                        {"nconmax": 1}, {"njmax": 1}, {"backend": "newton_cuda", "device": "cpu"}):
            with self.subTest(invalid=invalid):
                row = workflow.run_case(invalid)
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["samples"], [])
                self.assertEqual(row["diagnostics"]["error_type"], "ValueError")

    def test_full_native_tasks_both_robots_reset_after_full_warmup(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                row = workflow.run_case({"backend": "mujoco", "robot": robot, "warmups": 1, "repeats": 2})
                self.assertEqual(row["status"], "passed", row.get("reason"))
                self.assertEqual(row["scope"], "workflow")
                self.assertEqual(row["worlds"], 1)
                self.assertEqual(len(row["samples"]), 2)
                warmups = row["warmup_samples"]
                self.assertEqual(len(warmups), 1)
                self.assertEqual(len(row["timings"]["episode_setup_seconds"]), 3)
                self.assertEqual(len(row["workload"]["model_sha256"]), 64)
                self.assertIsNone(row["workload"]["controls_sha256"])
                self.assertIn("reconstructs", row["workload"]["model_equivalence"])
                for sample in warmups + row["samples"]:
                    self.assertTrue(sample["passed"])
                    self.assertEqual(sample["task_success_count"], 1)
                    self.assertEqual(sample["task_total_count"], 1)
                    for key in ("simulation_seconds", "output_transfer_seconds", "validation_seconds"):
                        self.assertGreater(sample[key], 0)
                    # The full warmup cannot leave a completed controller or a
                    # stacked cube behind for the measured episodes.
                    np.testing.assert_allclose(sample["task_metrics"]["red_xyz"],
                                               warmups[0]["task_metrics"]["red_xyz"], atol=1e-12)
                json.dumps(row, allow_nan=False)

    def test_failed_physical_task_stops_before_measured_repeats(self):
        def never_moves(controller, model, data, frame_dt):
            return controller.spec.home_ctrl.copy()

        with mock.patch.object(task.PickPlaceController, "step", never_moves):
            row = workflow.run_case({"backend": "mujoco", "repeats": 1, "warmups": 1})
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["samples"], [])
        self.assertEqual(row["warmup_samples"][0]["task_success_count"], 0)
        self.assertFalse(row["warmup_samples"][0]["passed"])
        self.assertIn("incomplete", row["reason"])

    def test_nonfinite_state_and_native_warnings_fail_even_with_good_final_positions(self):
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom type='sphere' size='.1'/></body></worldbody></mujoco>")
        data = mujoco.MjData(model)
        controller = SimpleNamespace(done=True)
        red = np.array([0.0, 0.0, 0.088])
        blue = np.array([0.0, 0.0, 0.044])
        data.qvel[0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            workflow._validate_native(mujoco, np, model, data, controller, red, blue)
        data.qvel[0] = 0.0
        data.warning[mujoco.mjtWarning.mjWARN_BADQVEL].number = 1
        with self.assertRaisesRegex(ValueError, "warnings"):
            workflow._validate_native(mujoco, np, model, data, controller, red, blue)

    def test_unstacked_cube_and_nonfinite_cube_cannot_pass(self):
        blue = np.array([0.0, 0.0, 0.044])
        for red in (np.array([0.0, 0.0, 0.044]), np.array([0.02, 0.0, 0.088]),
                    np.array([np.nan, 0.0, 0.088])):
            with self.subTest(red=red), self.assertRaises(ValueError):
                workflow._stack_metrics(np, red, blue)

    def test_unavailable_cuda_does_not_fall_back_to_cpu(self):
        import warp as wp

        with mock.patch.object(wp, "is_cuda_available", return_value=False), \
                mock.patch.object(task, "resolve_pick_place_scene") as resolve:
            row = workflow.run_case({"backend": "newton_cuda", "repeats": 1, "warmups": 1})
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(row["samples"], [])
        self.assertIn("No CUDA device", row["reason"])
        resolve.assert_not_called()

    def test_good_final_stack_does_not_hide_dragging_a_fall_or_unsettled_motion(self):
        observed = []
        validate = workflow.validate_payload_history

        def capture(history, spec, phases):
            observed.append((history.copy(), spec, phases.copy()))
            return validate(history, spec, phases)

        with mock.patch.object(workflow, "validate_payload_history", side_effect=capture):
            row = workflow.run_case({"backend": "mujoco", "robot": "rebot", "repeats": 1, "warmups": 1})
        self.assertEqual(row["status"], "passed", row.get("reason"))
        original, spec, phases = observed[-1]
        for failure in ("dragged", "fell_off_table", "moving", "nonfinite_history", "stalled_clock"):
            with self.subTest(failure=failure):
                changed = original.copy()
                if failure == "dragged":
                    changed[:500, 3] = spec.cube_center_z
                elif failure == "fell_off_table":
                    changed[-50:, 3] -= spec.table_top_z
                    changed[-50:, 10] -= spec.table_top_z
                elif failure == "moving":
                    changed[-50:, 15] = 0.2
                elif failure == "nonfinite_history":
                    changed[100, 3] = np.nan
                else:
                    changed[100:, 0] = changed[99, 0]
                self.assertFalse(validate(changed, spec, phases)["passed"])

    def test_device_recorder_converts_quaternion_and_preserves_payload_velocity(self):
        import warp as wp

        poses = np.array([[1, 2, 3, 0.0, 0.0, 0.6, 0.8], [4, 5, 6, 0.0, 0.6, 0.0, 0.8]], dtype=np.float32)
        velocities = np.arange(12, dtype=np.float32).reshape(2, 6)
        with wp.ScopedDevice("cpu"):
            body_q = wp.array(poses, dtype=wp.transform)
            body_qd = wp.array(velocities, dtype=wp.spatial_vector)
            sim_time = wp.array([0.02], dtype=float)
            history = wp.zeros((1, 27), dtype=float)
            wp.launch(workflow._record_payload_frame, dim=2,
                      inputs=[body_q, body_qd, sim_time, 1, 0, 0, history])
            result = history.numpy()[0]
        expected = np.concatenate(([0.02], poses[1, [0, 1, 2, 6, 3, 4, 5]],
                                   poses[0, [0, 1, 2, 6, 3, 4, 5]], velocities[1], velocities[0]))
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
