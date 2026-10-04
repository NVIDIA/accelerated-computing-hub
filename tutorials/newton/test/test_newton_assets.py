"""Pinned official USD imports; short native CPU probes, not task rollouts.

With usd-core 26.3, run with PXR_WORK_THREAD_LIMIT=1 before starting Python:
native parallel physics-parser crashes were observed on macOS and Linux.
The helper requires this limit on every platform instead of masking crashes.
"""
from pathlib import Path
import sys
import unittest
from unittest import mock
import warnings

import mujoco
import numpy as np
import newton
import warp as wp
from newton.solvers import SolverMuJoCo
from newton.utils import download_asset

PART = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"
sys.path.insert(0, str(PART))


class NewtonAssetsTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(newton, "use_coord_layout_targets", True))
        self.enterContext(warnings.catch_warnings())
        warnings.simplefilter("error")  # Never hide lost schema/constraint warnings.

    def test_so101_imports_official_joint_labels_and_passive_damping(self):
        from newton_assets import add_robot_usd
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = add_robot_usd(builder, "so101")
        expected = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.assertEqual([label.rsplit("/", 1)[-1] for label in builder.joint_label], expected)
        self.assertEqual(builder.body_count, 6)
        self.assertEqual(set(result["path_joint_map"].values()), set(range(6)))
        np.testing.assert_allclose(builder.joint_damping, [0.6] * 6)

    def test_rebot_preserves_mimic_in_native_cpu_steps(self):
        from newton_assets import add_robot_usd

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = add_robot_usd(builder, "rebot")
        expected = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint_left", "joint_right"]
        self.assertEqual([label.rsplit("/", 1)[-1] for label in builder.joint_label], expected)
        self.assertEqual(builder.body_count, 8)
        self.assertEqual(set(result["path_joint_map"].values()), set(range(8)))
        np.testing.assert_allclose(builder.joint_damping, [5, 5, 5, 2, 2, 2, 1, 1])
        self.assertEqual(builder.constraint_mimic_joint0, [6])
        self.assertEqual(builder.constraint_mimic_joint1, [7])
        self.assertEqual(builder.constraint_mimic_coef0, [0.0])
        self.assertEqual(builder.constraint_mimic_coef1, [1.0])
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True)
        self.assertEqual(solver.mj_model.nu, 8)
        self.assertEqual(solver.mj_model.neq, 1)
        np.testing.assert_array_equal(solver.mj_model.eq_type, [mujoco.mjtEq.mjEQ_JOINT])
        np.testing.assert_array_equal(solver.mj_model.eq_active0, [True])
        np.testing.assert_array_equal(solver.mj_model.eq_data[0, :5], [0, 1, 0, 0, 0])
        state, next_state = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        control = model.control()
        targets = control.joint_target_q.numpy().copy()
        target_indices = model.joint_target_q_start.numpy()[[6, 7]]
        targets[target_indices] = 0.01
        control.joint_target_q.assign(targets)
        for _ in range(12):
            state.clear_forces()
            solver.step(state, next_state, control, None, 0.001)
            state, next_state = next_state, state
            for field in (state.body_q, state.body_qd, state.joint_q, state.joint_qd):
                self.assertTrue(np.isfinite(field.numpy()).all())
            self.assertLess(abs(float(state.joint_q.numpy()[6] - state.joint_q.numpy()[7])), 5e-4)
        self.assertGreater(float(state.joint_q.numpy()[6]), 1e-5)
        self.assertTrue(all(w.number == 0 for w in solver.mj_data.warning))

    def test_download_uses_each_official_entry_at_the_literal_pin(self):
        from newton_assets import add_robot_usd

        for robot, folder, entry in (
            ("so101", "robotstudio_so101", "so101.usda"),
            ("rebot", "seeed_rebot_devarm", "seeed_rebot_devarm.usda"),
        ):
            with self.subTest(robot=robot):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                with mock.patch("newton_assets.download_asset", wraps=download_asset) as download, \
                     mock.patch.object(builder, "add_usd", wraps=builder.add_usd) as import_usd:
                    result = add_robot_usd(builder, robot)
                download.assert_called_once_with(folder, ref="f8fb7abcbeba2318814a74f3eeb02780ad7925d6")
                source = Path(import_usd.call_args.args[0])
                self.assertEqual(source.parts[-3:], (folder, "usd_structured", entry))
                self.assertIn("path_body_map", result)
                self.assertFalse(any(j == newton.JointType.FREE for j in builder.joint_type))
                # This field counts Newton external actuators, not MuJoCo USD actuators.
                self.assertEqual(result["actuator_count"], 0)
                self.assertEqual(len(builder.custom_attributes["mujoco:actuator_label"].values), builder.joint_count)

    def test_missing_registration_fails_before_download_or_mutation(self):
        from newton_assets import add_robot_usd

        builder = newton.ModelBuilder()
        with mock.patch("newton_assets.download_asset") as download:
            with self.assertRaisesRegex(RuntimeError, "SolverMuJoCo.register_custom_attributes"):
                add_robot_usd(builder, "rebot")
            download.assert_not_called()
        self.assertEqual((builder.body_count, builder.joint_count, builder.shape_count), (0, 0, 0))

    def test_unknown_robot_fails_before_download_or_mutation(self):
        from newton_assets import add_robot_usd

        builder = newton.ModelBuilder()
        with mock.patch("newton_assets.download_asset") as download:
            with self.assertRaisesRegex(ValueError, "Unknown robot"):
                add_robot_usd(builder, "SO101")
            download.assert_not_called()
        self.assertEqual((builder.body_count, builder.joint_count, builder.shape_count), (0, 0, 0))

    def test_download_failure_propagates_without_a_substitute_asset(self):
        from newton_assets import add_robot_usd

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        failure = ConnectionError("official asset download failed")
        with mock.patch("newton_assets.download_asset", side_effect=failure) as download, \
             mock.patch.object(builder, "add_usd") as import_usd, \
             mock.patch.object(builder, "add_mjcf") as import_mjcf:
            with self.assertRaises(ConnectionError) as caught:
                add_robot_usd(builder, "rebot")
            self.assertIs(caught.exception, failure)
            download.assert_called_once()
            import_usd.assert_not_called()
            import_mjcf.assert_not_called()
        self.assertEqual((builder.body_count, builder.joint_count, builder.shape_count), (0, 0, 0))

    def test_unverified_cached_usd_is_rejected_before_import(self):
        from newton_assets import add_robot_usd

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        failure = RuntimeError("Unverified asset cache: tracked robot files are modified")
        with mock.patch("newton_assets.download_asset", return_value=Path("cache/robotstudio_so101")), \
             mock.patch("newton_assets.validate_pinned_asset_cache", side_effect=failure), \
             mock.patch.object(builder, "add_usd") as import_usd:
            with self.assertRaises(RuntimeError) as caught:
                add_robot_usd(builder, "so101")
            self.assertIs(caught.exception, failure)
            import_usd.assert_not_called()
        self.assertEqual((builder.body_count, builder.joint_count, builder.shape_count), (0, 0, 0))

    def test_unsafe_usd263_parser_configuration_fails_before_loading_on_every_platform(self):
        from pxr import Usd, Work
        from newton_assets import add_robot_usd

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        for platform in ("darwin", "linux", "win32"):
            with self.subTest(platform=platform), \
                 mock.patch.object(sys, "platform", platform), \
                 mock.patch.object(Usd, "GetVersion", return_value=(0, 26, 3)), \
                 mock.patch.object(Work, "GetConcurrencyLimit", return_value=12), \
                 mock.patch.object(Work, "SetConcurrencyLimit") as set_limit, \
                 mock.patch("newton_assets.download_asset", return_value=Path("unused")) as download, \
                 mock.patch.object(builder, "add_usd", return_value={}) as import_usd:
                with self.assertRaisesRegex(RuntimeError, "PXR_WORK_THREAD_LIMIT=1 before starting Python"):
                    add_robot_usd(builder, "rebot")
                download.assert_not_called()
                import_usd.assert_not_called()
                set_limit.assert_not_called()

    def test_usd_parser_guard_allows_serial_imports_and_does_not_restrict_other_versions(self):
        from pxr import Usd, Work
        from newton_assets import add_robot_usd

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        for version, concurrency in (((0, 26, 3), 1), ((0, 26, 5), 12)):
            with self.subTest(version=version, concurrency=concurrency), \
                 mock.patch.object(Usd, "GetVersion", return_value=version), \
                 mock.patch.object(Work, "GetConcurrencyLimit", return_value=concurrency), \
                 mock.patch.object(Work, "SetConcurrencyLimit") as set_limit, \
                 mock.patch("newton_assets.download_asset", return_value=Path("unused")) as download, \
                 mock.patch("newton_assets.validate_pinned_asset_cache"), \
                 mock.patch.object(builder, "add_usd", return_value={}) as import_usd:
                self.assertEqual(add_robot_usd(builder, "rebot"), {})
                download.assert_called_once()
                import_usd.assert_called_once()
                set_limit.assert_not_called()

    def test_both_robots_drive_native_cpu_targets_after_a_free_body(self):
        from newton_assets import add_robot_usd

        for robot, count, force in (("so101", 6, [2.94] * 6), ("rebot", 8, [36, 36, 36, 14, 14, 14, 1904, 1904])):
            with self.subTest(robot=robot):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                prefix = builder.add_body(xform=wp.transform((2.0, 0.0, 0.5), wp.quat_identity()), label="prefix")
                builder.add_shape_box(body=prefix, hx=0.02, hy=0.02, hz=0.02,
                                      cfg=newton.ModelBuilder.ShapeConfig(gap=0.0))
                result = add_robot_usd(builder, robot)
                # Imported paths map into this builder, not an assumed standalone 0..N.
                target_labels = builder.custom_attributes["mujoco:actuator_target_label"].values
                joints = [result["path_joint_map"][label] for label in target_labels]
                self.assertEqual(len(joints), count)
                self.assertTrue(all(j > 0 for j in joints))
                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True)
                self.assertTrue(solver.use_mujoco_cpu)
                self.assertEqual(solver.mj_model.nu, count)  # No duplicate drive actuators.
                indices = model.joint_target_q_start.numpy()[joints]
                self.assertTrue(np.all(indices != model.joint_qd_start.numpy()[joints]))
                np.testing.assert_array_equal(solver.mjc_actuator_to_newton_target_q_idx.numpy(), indices)
                np.testing.assert_array_equal(solver.mjc_actuator_ctrl_source.numpy(), np.zeros(count))
                np.testing.assert_allclose(solver.mj_model.actuator_forcerange, np.column_stack((-np.array(force), force)))
                np.testing.assert_allclose(solver.mj_model.actuator_ctrlrange,
                                           np.asarray(builder.custom_attributes["mujoco:actuator_ctrlrange"].values))
                np.testing.assert_allclose(solver.mj_model.dof_damping[6:], builder.joint_damping[6:])
                colliding = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
                np.testing.assert_array_equal(model.shape_gap.numpy()[colliding], 0.0)
                state, next_state = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                targets = control.joint_target_q.numpy().copy()
                expected = np.zeros(count)
                expected[0] = 0.02
                targets[indices] = expected
                control.joint_target_q.assign(targets)
                for _ in range(12):
                    state.clear_forces()
                    solver.step(state, next_state, control, None, 0.001)
                    state, next_state = next_state, state
                    np.testing.assert_allclose(solver.mj_data.ctrl, expected)
                    for field in (state.body_q, state.body_qd, state.joint_q, state.joint_qd):
                        self.assertTrue(np.isfinite(field.numpy()).all())
                self.assertTrue(all(w.number == 0 for w in solver.mj_data.warning))

    def test_rebot_preserves_nonadjacent_authored_collision_exclusion(self):
        from newton_assets import add_robot_usd

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = add_robot_usd(builder, "rebot")
        body_map = {path.rsplit("/", 1)[-1]: index for path, index in result["path_body_map"].items()}
        self.assertEqual(body_map["base_link"], -1)
        self.assertEqual(body_map["gripper_end"], body_map["link6"])
        filters = set(builder.shape_collision_filter_pairs)
        colliders = lambda body: [i for i, b in enumerate(builder.shape_body) if b == body
                                  and builder.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)]
        first, second = colliders(body_map["link2"]), colliders(body_map["link5"])
        self.assertTrue(first and second)
        self.assertTrue(all(tuple(sorted((a, b))) in filters for a in first for b in second))
        # Importing exclusions must not turn off every possible robot self-contact.
        other = colliders(body_map["gripper_right"])
        self.assertTrue(other)
        self.assertTrue(any(tuple(sorted((a, b))) not in filters for a in first for b in other))
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True)
        # SolverMuJoCo sanitizes USD paths in names; use its public ID mapping.
        native_to_newton = solver.mjc_body_to_newton.numpy()[0]
        mj_bodies = [int(np.flatnonzero(native_to_newton == body_map[name])[0])
                     for name in ("link2", "link5")]
        self.assertTrue(all(i > 0 for i in mj_bodies))
        first_id, second_id = sorted(mj_bodies)
        self.assertIn((first_id << 16) + second_id, solver.mj_model.exclude_signature)

    def test_xform_moves_bodies_and_collapsed_base_shapes(self):
        from newton_assets import add_robot_usd

        pose = wp.transform((0.1, -0.2, 0.3), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4))
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                models, states = [], []
                for xform in (None, pose):
                    builder = newton.ModelBuilder()
                    SolverMuJoCo.register_custom_attributes(builder)
                    result = add_robot_usd(builder, robot, xform=xform)
                    if robot == "so101":
                        body_map = {p.rsplit("/", 1)[-1]: i for p, i in result["path_body_map"].items()}
                        self.assertEqual(body_map["base"], -1)
                        self.assertEqual(body_map["camera_mount"], body_map["gripper"])
                    model = builder.finalize(device="cpu")
                    state = model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                    models.append(model)
                    states.append(state)
                for original, moved in zip(states[0].body_q.numpy(), states[1].body_q.numpy()):
                    transformed = pose * wp.transform(*original)
                    np.testing.assert_allclose(moved[:3], wp.transform_get_translation(transformed), atol=2e-6)
                    np.testing.assert_allclose(wp.quat_to_matrix(wp.quat(*moved[3:])),
                                               wp.quat_to_matrix(wp.transform_get_rotation(transformed)), atol=2e-6)
                static = np.flatnonzero(models[0].shape_body.numpy() == -1)
                self.assertGreater(len(static), 0)
                for shape in static:
                    transformed = pose * wp.transform(*models[0].shape_transform.numpy()[shape])
                    np.testing.assert_allclose(models[1].shape_transform.numpy()[shape, :3],
                                               wp.transform_get_translation(transformed), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
