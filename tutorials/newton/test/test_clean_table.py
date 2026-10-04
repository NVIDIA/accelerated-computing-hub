"""Behavioral checks for the Newton clean-table reference (no viewer needed)."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest import mock

import numpy as np

PART = (Path(__file__).resolve().parents[1] / "notebooks") / "newton" / "part3"
sys.path.insert(0, str(PART))
sys.path.insert(0, str(PART / "solutions"))


def contained_state_fixture():
    """Checker-only fixture; hand-placed State is NOT a physical rollout."""
    from clean_the_table_solution import Example

    example = Example()
    scene, state = example.scene, example.state_0
    center = (scene.bin.lower + scene.bin.upper) / 2
    points = scene.payload_points(state)
    q = state.body_q.numpy()
    for name, shape in scene.cube_shapes.items():
        body = scene.model.shape_body.numpy()[shape]
        q[body, :3] += center - points[name].mean(axis=0)
    q[scene.cable_bodies, :3] += center - points["cable"].mean(axis=0)
    state.particle_q.assign(state.particle_q.numpy() + center - points["shirt"].mean(axis=0))
    # Deliberately move both jaws into the receiving box as well. The fixture
    # supplies no physical grasp history, regardless of these terminal poses.
    shift = center - scene.jaw_points(state).mean(axis=0)
    for body in set(scene.jaw_bodies):
        q[body, :3] += shift
    state.body_q.assign(q)
    state.body_qd.zero_()
    state.particle_qd.zero_()
    example.phase = "done"
    # Deliberate protocol preconditions, not measured feedback from this fixture.
    example.max_coupling_force = 0.2
    example.max_soft_contacts = 3
    assert all(scene.bin.contains(p) for p in scene.payload_points(state).values())
    assert all(speed == 0 for speed in scene.payload_speeds(state).values())
    observe_fixture_frame(example)
    return example


def observe_fixture_frame(example):
    """Advance only the checker fixture's observation clock, never physics."""
    example.frames += 1
    example.sim_time = example.frames * example.frame_dt
    example._observe()


class CleanTableSceneTests(unittest.TestCase):
    def test_mujoco_capacity_guard_rejects_sticky_overflow_but_retains_convergence_flags(self):
        import mujoco_warp as mjw
        import warp as wp
        from clean_table_task import check_mujoco_capacity

        self.assertEqual(check_mujoco_capacity(SimpleNamespace()), 0)
        convergence = int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
        flags = wp.array([0, convergence], dtype=wp.int32, device="cpu")
        solver = SimpleNamespace(mjw_data=SimpleNamespace(overflow=flags))
        self.assertEqual(check_mujoco_capacity(solver), convergence)
        for overflow in (mjw.OverflowType.NEFC, mjw.OverflowType.NARROWPHASE):
            flags.assign(np.asarray([int(overflow), convergence], dtype=np.int32))
            with self.subTest(overflow=overflow), self.assertRaisesRegex(ValueError, "capacity overflow"):
                check_mujoco_capacity(solver)

    def test_checked_runner_stops_on_the_goal_and_fails_at_the_frame_budget(self):
        from newton.viewer import ViewerNull
        from clean_the_table_solution import run_checked

        class CounterTask:
            """Loop-control fixture, deliberately NOT physical task evidence."""
            def __init__(self, ready_at):
                self.viewer = ViewerNull(num_frames=100)
                self.frames = 0
                self.ready_at = ready_at
                self.final_checked = False
                self.failure = None
                self.phase = "running"

            def step(self):
                self.frames += 1

            def render(self):
                self.viewer.begin_frame(float(self.frames))
                self.viewer.end_frame()

            def is_complete(self):
                return self.frames >= self.ready_at

            def test_final(self):
                self.final_checked = True
                if not self.is_complete():
                    raise ValueError("physical goal not achieved within the budget")

        for ready_at, frames in ((3, 3), (8, 5)):
            task = CounterTask(ready_at)
            with self.subTest(ready_at=ready_at), mock.patch.object(
                task.viewer, "close", wraps=task.viewer.close,
            ) as close:
                if ready_at > 5:
                    with self.assertRaisesRegex(ValueError, "goal not achieved"):
                        run_checked(task, max_frames=5)
                else:
                    run_checked(task, max_frames=5)
                self.assertEqual(task.frames, frames)
                self.assertEqual(task.viewer.frame_count, frames)
                self.assertTrue(task.final_checked)
                close.assert_called_once()

    def test_unearned_terminal_phase_cannot_spin_without_advancing_the_frame_budget(self):
        from newton.viewer import ViewerNull
        from clean_the_table_solution import run_checked

        class UnearnedTerminalTask:
            """Bounded loop fixture; no physical task outcome is asserted."""
            def __init__(self):
                self.viewer = ViewerNull(num_frames=4)
                self.frames, self.steps = 0, 0
                self.failure, self.phase = None, "done"

            def is_complete(self):
                return False

            def step(self):
                self.steps += 1  # A terminal controller has no next physics step.

            def render(self):
                self.viewer.begin_frame(0.)
                self.viewer.end_frame()

            def test_final(self):
                raise ValueError("unearned terminal state")

        example = UnearnedTerminalTask()
        with self.assertRaises(ValueError):
            run_checked(example, max_frames=5)
        self.assertLessEqual(example.steps, 1, "An unearned terminal phase must fail without spinning")

    def test_done_transition_avoids_extra_physics_and_aligns_terminal_recording_labels(self):
        """A controller/recorder fixture, deliberately lacking grasp history."""
        from clean_the_table_solution import Example

        for terminal_pose_already_recorded in (False, True):
            with self.subTest(terminal_pose_already_recorded=terminal_pose_already_recorded):
                example = Example()
                example.phase = "final_settle"
                example.object_index = len(example.targets)
                observe_fixture_frame(example)
                if terminal_pose_already_recorded:
                    example._record_frame()
                frame, sim_time = example.frames, example.sim_time
                observations = len(example.observations)
                q = example.state_0.body_q.numpy().copy()

                def finish_controller():
                    example.phase = "done"

                with mock.patch.object(example, "_advance_phase", side_effect=finish_controller), \
                     mock.patch.object(example, "simulate") as simulate, \
                     mock.patch.object(example.ik, "solve") as solve:
                    example.step()
                simulate.assert_not_called()
                solve.assert_not_called()
                self.assertEqual((example.frames, example.sim_time), (frame, sim_time))
                self.assertEqual(len(example.observations), observations)
                self.assertEqual(example.observations[-1]["phase"], "done")
                self.assertEqual(example.observations[-1]["active_object"], "")
                np.testing.assert_array_equal(example.state_0.body_q.numpy(), q)
                with tempfile.TemporaryDirectory() as directory:
                    record = Path(directory) / "terminal-label-fixture.npz"
                    example.args = SimpleNamespace(record=str(record))
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "incomplete"):
                        example.test_final()
                    with np.load(record, allow_pickle=False) as data:
                        self.assertTrue(np.all(np.diff(data["time"]) > 0))
                        self.assertEqual(data["time"][-1], sim_time)
                        self.assertEqual(data["observation_time"][-1], sim_time)
                        self.assertEqual(data["phase"][-1], "done")
                        self.assertEqual(data["observation_phase"][-1], "done")
                        self.assertEqual(data["active_object"][-1], "")
                        self.assertEqual(data["observation_active_object"][-1], "")
                        np.testing.assert_array_equal(data["body_q"][-1], q)

    def test_advanced_robot_physics_imports_the_pinned_official_usd(self):
        import newton_assets
        from clean_table_scene_solution import build_scene

        for robot, folder in (("so101", "robotstudio_so101"), ("rebot", "seeed_rebot_devarm")):
            with self.subTest(robot=robot), mock.patch.object(
                newton_assets, "download_asset", wraps=newton_assets.download_asset,
            ) as download:
                scene = build_scene(robot, device="cpu")
                download.assert_called_once_with(folder, ref=newton_assets.NEWTON_ASSETS_REF)
                cube_bodies = {int(scene.model.shape_body.numpy()[shape]) for shape in scene.cube_shapes.values()}
                robot_bodies = set(scene.rigid_bodies) - cube_bodies
                self.assertTrue(robot_bodies)
                self.assertTrue(all(scene.model.body_label[body].startswith("/") for body in robot_bodies))
                self.assertTrue(cube_bodies.issubset(scene.rigid_bodies))

    def test_scene_uses_current_rod_and_collision_apis_without_deprecations(self):
        import newton
        from clean_table_scene_solution import build_coupled_solver, build_scene

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            scene = build_scene("so101", device="cpu")
            build_coupled_solver(scene)
        types = scene.model.joint_type.numpy()[scene.cable_joints]
        self.assertTrue(np.all(types == int(newton.JointType.ROD)))

    def test_report_identifies_the_usd_configuration_not_the_mjcf_baseline(self):
        from clean_the_table_solution import Example

        report = Example().report()
        self.assertEqual(report.get("robot_asset_format"), "USD")
        self.assertEqual(report.get("newton_assets_ref"), "f8fb7abcbeba2318814a74f3eeb02780ad7925d6")

    def test_compact_material_contact_shapes_resolve_to_the_global_body(self):
        from clean_the_table_solution import Example

        for robot in ("so101", "rebot"):
            example = Example(args=SimpleNamespace(robot=robot, device="cpu"))
            view = example.solver.view("deformable")
            global_shape_ids = {label: i for i, label in enumerate(example.model.shape_label)}
            expected = np.asarray([
                example._shape_body[global_shape_ids[label]] for label in view.shape_label
            ])
            # Shapes of unowned upper-arm bodies remain in the view but are
            # disabled and have no destination body; they must stay unmapped.
            expected[view.shape_body.numpy() < 0] = -1
            with self.subTest(robot=robot):
                np.testing.assert_array_equal(example._vbd_shape_body, expected)
                self.assertTrue(set(example.scene.jaw_bodies).issubset(expected))
                self.assertTrue(set(example.scene.cable_bodies).issubset(expected))
                # Ensure this actually exercises compaction, not an identity map.
                self.assertTrue(np.any(view.shape_body.numpy() != expected))

    def test_material_force_telemetry_separates_cable_and_cloth_reactions(self):
        """Synthetic contact buffers verify attribution; they are not a rollout."""
        import warp as wp
        from clean_the_table_solution import Example

        example = Example()
        jaws = example.scene.jaw_bodies
        local_jaws = [int(np.flatnonzero(example._vbd_shape_body == body)[0]) for body in jaws]
        local_cable = int(np.flatnonzero(example._vbd_shape_body == example.scene.cable_bodies[0])[0])
        axis_points = example.scene.jaw_points(example.state_0)
        axis = axis_points[1] - axis_points[0]
        axis /= np.linalg.norm(axis)
        vbd = example.solver.solver("deformable")
        for reverse in (False, True):
            for cloth_force in (0., .3):
                # These are real compact shape IDs, with known test forces.
                reaction = np.asarray([-axis * .7, axis * .7])
                raw = np.zeros((example.model.body_count, 6))
                raw[list(jaws), :3] = reaction * ((.7 + cloth_force) / .7)
                vbd.last_raw_body_f = wp.array(raw, dtype=wp.spatial_vector, device="cpu")
                contacts = SimpleNamespace(
                    rigid_contact_count=wp.array([2], dtype=int, device="cpu"),
                    rigid_contact_shape0=wp.array(local_jaws if reverse else [local_cable] * 2, dtype=int, device="cpu"),
                    rigid_contact_shape1=wp.array([local_cable] * 2 if reverse else local_jaws, dtype=int, device="cpu"),
                    rigid_contact_force=wp.array(-reaction if reverse else reaction, dtype=wp.vec3, device="cpu"),
                    soft_contact_count=wp.array([2], dtype=int, device="cpu"),
                    soft_contact_shape=wp.array(local_jaws, dtype=int, device="cpu"),
                )
                for name in example.initial_points:
                    example._frame_contacts[name].fill(0)
                    example._frame_forces[name].fill(0)
                with self.subTest(reverse=reverse, cloth_force=cloth_force), mock.patch.object(
                    example.solver, "get_proxy_contacts", return_value=contacts,
                ):
                    example._sample_contacts()
                    np.testing.assert_allclose(example._frame_forces["cable"], [.7, .7], atol=1e-7)
                    np.testing.assert_allclose(example._frame_forces["shirt"], [cloth_force] * 2, atol=1e-7)
                    np.testing.assert_array_equal(example._frame_contacts["cable"], [1, 1])
                    np.testing.assert_array_equal(example._frame_forces["red_cube"], [0., 0.])
                    np.testing.assert_array_equal(example._frame_forces["blue_cube"], [0., 0.])

    def test_usd_jaw_poses_match_cube_and_material_controller_sites_for_both_robots(self):
        import newton
        import warp as wp
        from clean_table_scene_solution import build_scene
        from clean_table_task import GripperIK

        for robot in ("so101", "rebot"):
            scene = build_scene(robot, device="cpu")
            state = scene.model.state()
            ik = GripperIK(scene.ik_model, scene.spec)
            ik.ctrl = scene.initial_ctrl.copy()
            for material in (False, True):
                origin = scene.jaw_points(state, material=material).mean(axis=0)
                for offset in ([0., 0., 0.], [-.01, .01, .01], [.01, .015, .005]):
                    target = origin + offset
                    ctrl = ik.solve(target, scene.spec.gripper_open, material=material, iterations=250)
                    q = scene.model.joint_q.numpy().copy()
                    q[scene.robot_q_indices] = ctrl
                    newton.eval_fk(scene.model, wp.array(q, dtype=wp.float32, device="cpu"), scene.model.joint_qd, state)
                    measured = scene.jaw_points(state, material=material)
                    with self.subTest(robot=robot, material=material, target=target.tolist()):
                        np.testing.assert_allclose(measured, ik.jaw_points(), atol=2e-5)
                        self.assertLess(np.linalg.norm(measured.mean(axis=0) - target), 0.001)

    def test_record_hash_binds_the_new_npz_before_the_report_is_written(self):
        example = contained_state_fixture()
        example.history_times = [example.sim_time]
        example.history_body = [example.state_0.body_q.numpy().copy()]
        example.history_particle = [example.state_0.particle_q.numpy().copy()]
        with tempfile.TemporaryDirectory() as directory:
            record, report_path = Path(directory) / "fixture.npz", Path(directory) / "fixture.json"
            record.write_bytes(b"stale recording from a previous attempt")
            example.args = SimpleNamespace(record=str(record), report=str(report_path))
            write_text = Path.write_text

            def write_report(path, text, *args, **kwargs):
                report = json.loads(text)
                self.assertNotEqual(record.read_bytes(), b"stale recording from a previous attempt",
                                    "Save the new recording BEFORE writing its report")
                self.assertEqual(report.get("record_sha256"), hashlib.sha256(record.read_bytes()).hexdigest())
                return write_text(path, text, *args, **kwargs)

            log = io.StringIO()
            with mock.patch.object(Path, "write_text", autospec=True, side_effect=write_report), \
                 contextlib.redirect_stdout(log), self.assertRaisesRegex(ValueError, "incomplete"):
                example.test_final()
            report = json.loads(report_path.read_text())
            # This deliberately unsuccessful checker fixture is not evidence of a rollout.
            self.assertIs(report["success"], False)
            self.assertEqual(report["record_sha256"], hashlib.sha256(record.read_bytes()).hexdigest())
            printed = next(line.removeprefix("CLEAN_TABLE_RESULT ") for line in log.getvalue().splitlines()
                           if line.startswith("CLEAN_TABLE_RESULT "))
            self.assertEqual(json.loads(printed), report)
            with np.load(record) as data:
                np.testing.assert_array_equal(data["body_q"][-1], example.state_0.body_q.numpy())
            # Keep NumPy's existing extensionless-path behavior when hashing.
            record.unlink()
            example.args.record = str(record.with_suffix(""))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                example.test_final()
            self.assertFalse(record.with_suffix("").exists())
            self.assertEqual(json.loads(report_path.read_text())["record_sha256"],
                             hashlib.sha256(record.read_bytes()).hexdigest())

    def test_reference_baseline_accepts_real_partial_progress_after_step_zero(self):
        students = {PART / name: (PART / name).read_bytes()
                    for name in ("clean_table_scene.py", "clean_the_table.py")}
        source = PART / "clean_table_scene.py"
        namespace = {"__file__": str(source)}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        snippet = PART / "solutions" / "clean_step_00_add_bin.py"
        exec(compile(snippet.read_text(), str(snippet), "exec"), namespace)
        # Exercise the real build order without editing either student file.
        with self.assertRaisesRegex(NotImplementedError, "^Complete TODO Step 2: add_cable$") as caught:
            namespace["build_scene"]("so101", device="cpu")
        notebook = json.loads((PART / "03__clean_the_table.ipynb").read_text())
        cell = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "clean-08")
        failure = subprocess.CalledProcessError(1, [sys.executable], output="",
                    stderr=f"{type(caught.exception).__name__}: {caught.exception}")
        with mock.patch.object(subprocess, "run", side_effect=failure), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell, "advanced-partial-baseline", "exec"),
                 {"PART": PART, "sys": sys, "subprocess": subprocess, "REFERENCE": True})
        for path, before in students.items():
            self.assertEqual(path.read_bytes(), before)

    def test_record_hash_is_absent_when_no_new_recording_was_saved(self):
        example = contained_state_fixture()
        with tempfile.TemporaryDirectory() as directory:
            record, report_path = Path(directory) / "fixture.npz", Path(directory) / "fixture.json"
            record.write_bytes(b"old file is not this run")
            example.args = SimpleNamespace(report=str(report_path), record=None)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                example.test_final()
            self.assertIsNone(json.loads(report_path.read_text())["record_sha256"])
            self.assertEqual(record.read_bytes(), b"old file is not this run")

    def test_recording_includes_the_actual_terminal_state_between_stride_frames(self):
        # A failed checker fixture, not a simulation or a publishable result.
        example = contained_state_fixture()
        example.history_times = [example.sim_time - example.frame_dt]
        example.history_body = [example.state_0.body_q.numpy().copy()]
        example.history_particle = [example.state_0.particle_q.numpy().copy()]
        q = example.state_0.body_q.numpy()
        q[example.scene.cable_bodies[0], 0] += 0.001
        example.state_0.body_q.assign(q)
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "terminal.npz"
            example.args = SimpleNamespace(record=str(record))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                example.test_final()
            with np.load(record) as data:
                self.assertEqual(len(data["time"]), 2)
                self.assertEqual(data["time"][-1], example.sim_time)
                np.testing.assert_array_equal(data["body_q"][-1], example.state_0.body_q.numpy())

    def test_unwithdrawn_jaws_cannot_pass_even_with_contained_still_payloads(self):
        example = contained_state_fixture()
        self.assertEqual(example.phase, "done")
        for _ in range(example.fps):
            observe_fixture_frame(example)
        with self.assertRaisesRegex(ValueError, "incomplete"), contextlib.redirect_stdout(io.StringIO()):
            example.test_final()
        self.assertTrue(all(ev.settled_frames == 0 for ev in example.evidence.values()))
        report = example.report()
        self.assertIs(report["tool_withdrawn"], False)
        self.assertLess(report["tool_clearance_m"], 0.0)
        # A stale settling counter cannot replace a final measured tool pose.
        for ev in example.evidence.values():
            ev.settled_frames = ev.detached_settled_frames = example.fps
        with self.assertRaisesRegex(ValueError, "incomplete"), contextlib.redirect_stdout(io.StringIO()):
            example.test_final()

    def test_manufactured_containment_and_withdrawal_never_replace_grasp_history(self):
        example = contained_state_fixture()
        scene = example.scene
        q = example.state_0.body_q.numpy()
        shift = .025 - scene.tool_clearance(example.state_0)
        for body in set(scene.jaw_bodies):
            q[body, 2] += shift
        example.state_0.body_q.assign(q)
        before = example.state_0.body_q.numpy().copy()
        for _ in range(example.fps):
            observe_fixture_frame(example)
        report = example.report()
        self.assertIs(report["tool_withdrawn"], True)
        self.assertAlmostEqual(report["tool_clearance_m"], 0.025, places=6)
        self.assertTrue(all(obj["inside"] for obj in report["objects"].values()))
        self.assertGreater(report["max_coupling_input_force_norm"], 0.)
        self.assertEqual(report["phase"], "done")
        self.assertFalse(example.is_complete())
        for ev in example.evidence.values():
            ev.settled_frames = ev.detached_settled_frames = example.fps
        self.assertFalse(example.is_complete(), "Stale settling counters cannot manufacture grasp history")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "incomplete"):
            example.test_final()
        np.testing.assert_array_equal(example.state_0.body_q.numpy(), before)

    def test_free_garment_starts_clear_of_the_rigid_clutter(self):
        from clean_table_scene_solution import build_scene

        scene = build_scene("so101", device="cpu")
        points = scene.payload_points(scene.model.state())
        cloth = points["shirt"]
        self.assertGreater(cloth[:, 2].min(), scene.table_z)
        for name in ("red_cube", "blue_cube"):
            cube = points[name]
            separated = np.any(cloth[:, :2].max(axis=0) < cube[:, :2].min(axis=0)) or np.any(
                cube[:, :2].max(axis=0) < cloth[:, :2].min(axis=0))
            self.assertTrue(separated, f"Initial cloth intersects {name} in the tabletop layout")

    def test_contact_gaps_are_explicit_at_manipulation_scale(self):
        from clean_table_scene_solution import build_scene
        import newton

        for robot in ("so101", "rebot"):
            scene = build_scene(robot, device="cpu")
            colliders = (scene.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
            self.assertLessEqual(float(scene.model.shape_gap.numpy()[colliders].max()), 0.005001)

    def test_untouched_cubes_remain_supported_in_the_native_rigid_solver(self):
        """Exercise real contacts for eight seconds, independently of VBD/IK."""
        import mujoco
        from clean_table_scene_solution import build_coupled_solver, build_scene

        for robot in ("so101", "rebot"):
            scene = build_scene(robot, device="cpu")
            solver = build_coupled_solver(scene).solver("rigid")
            model, data = solver.mj_model, mujoco.MjData(solver.mj_model)
            data.ctrl[:] = scene.initial_ctrl
            mujoco.mj_forward(model, data)
            bodies = [model.body(name).id for name in scene.cube_shapes]
            late_z = []
            steps = int(round(8. / model.opt.timestep))
            for step in range(steps):
                mujoco.mj_step(model, data)
                if step * model.opt.timestep >= 7.:
                    late_z.append(data.xpos[bodies, 2].copy())
            with self.subTest(robot=robot):
                self.assertTrue(np.isfinite(data.qpos).all())
                np.testing.assert_allclose(late_z, scene.table_z + .022, atol=.001)
                self.assertLess(float(np.max(np.abs(data.cvel[bodies]))), .001)

    def test_authored_cable_has_no_unfiltered_initial_self_contacts(self):
        import newton
        from clean_table_scene_solution import build_scene, make_clutter_pipeline

        scene = build_scene("so101", device="cpu")
        state = scene.model.state()
        newton.eval_fk(scene.model, scene.model.joint_q, scene.model.joint_qd, state)
        pipeline = make_clutter_pipeline(scene.model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        cable = set(scene.cable_shapes)
        self_pairs = [(int(a), int(b)) for a, b in zip(
            contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count],
        ) if int(a) in cable and int(b) in cable]
        self.assertEqual(self_pairs, [], "Overlapping non-neighbor capsules destabilize the free cable")

    def test_rebot_cable_grasp_follows_one_interior_segment_despite_height_ties(self):
        from clean_table_scene_solution import build_scene

        scene = build_scene("rebot", device="cpu")
        state = scene.model.state()
        bodies = scene.cable_bodies
        selected = bodies[(len(bodies) - 1) // 2]
        self.assertNotIn(selected, (bodies[0], bodies[-1]))
        poses = state.body_q.numpy().copy()
        # Deliberate geometry fixtures, not a simulated or accepted task state.
        # Settled capsules can trade height ranks while remaining almost level.
        poses[bodies, 2] = scene.table_z + .007
        target = poses[selected, :3].copy()
        for highest in (bodies[0], bodies[-1]):
            perturbed = poses.copy()
            perturbed[highest, 2] += 1e-6
            state.body_q.assign(perturbed)
            with self.subTest(highest_body=highest):
                self.assertEqual(bodies[int(np.argmax(perturbed[bodies, 2]))], highest)
                np.testing.assert_array_equal(scene.grasp_point(state, "cable"), target)
                np.testing.assert_array_equal(state.body_q.numpy(), perturbed)

        poses[selected, :3] += np.asarray([.004, -.002, .001])
        state.body_q.assign(poses)
        np.testing.assert_array_equal(scene.grasp_point(state, "cable"), poses[selected, :3])
        self.assertGreater(np.linalg.norm(poses[selected, :3] - target), .004)
        np.testing.assert_array_equal(state.body_q.numpy(), poses)

    def test_gripper_robot_preserves_the_reference_actuator_limits(self):
        from clean_table_scene_solution import build_coupled_solver, build_scene
        import pick_place_common as task

        scene = build_scene("so101", device="cpu")
        solver = build_coupled_solver(scene)
        reference = task.load_pick_place_model(task.resolve_pick_place_scene(spec=scene.spec), scene.spec)
        np.testing.assert_allclose(solver.solver("rigid").mj_model.actuator_forcerange, reference.actuator_forcerange)

    def test_scene_has_free_garment_cable_and_rigid_clutter_outside_bin(self):
        from clean_table_scene_solution import build_scene

        scene = build_scene("so101", device="cpu")
        state = scene.model.state()
        self.assertEqual(set(scene.payload_points(state)), {"red_cube", "blue_cube", "shirt", "cable"})
        self.assertGreater(scene.model.particle_count, 0)
        self.assertGreater(len(scene.cable_bodies), 2)
        self.assertFalse(set(scene.rigid_bodies) & set(scene.cable_bodies))
        self.assertFalse(set(scene.rigid_joints) & set(scene.cable_joints))
        self.assertTrue(np.all(scene.model.particle_mass.numpy() > 0))
        self.assertEqual(len(scene.bin_shapes), 5)
        for name, points in scene.payload_points(state).items():
            with self.subTest(payload=name):
                self.assertTrue(np.isfinite(points).all())
                self.assertFalse(scene.bin.contains(points))

    def test_simulation_rejects_an_unfinished_rollout(self):
        from clean_the_table_solution import Example

        example = Example()
        example.step()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            example.test_final()

    def test_nonfinite_joint_state_is_rejected_immediately(self):
        from clean_the_table_solution import Example

        example = Example()
        q = example.state_0.joint_q.numpy()
        q[0] = np.nan
        example.state_0.joint_q.assign(q)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            example._observe()

    def test_bin_rejects_partial_containment_and_nonfinite_geometry(self):
        from clean_table_task import Bin

        box = Bin(np.zeros(3), np.ones(3), tolerance=0.0)
        self.assertTrue(box.contains(np.asarray([[0.1, 0.5, 0.9]])))
        self.assertFalse(box.contains(np.asarray([[0.5, 0.5, 0.5], [1.01, 0.5, 0.5]])))
        self.assertFalse(box.contains(np.asarray([[np.nan, 0.5, 0.5]])))
        self.assertFalse(box.contains(np.empty((0, 3))))

    def test_contact_overflow_fails_before_the_solver_drops_contacts(self):
        from clean_table_task import validate_contact_counts

        shape_bodies = np.asarray([-1, 0, 1])
        validate_contact_counts(shape_bodies, np.asarray([[0, 0], [0, 0]]),
                                np.asarray([0, 0]), rigid_limit=1, soft_limit=1)
        with self.assertRaisesRegex(ValueError, "rigid contact"):
            validate_contact_counts(shape_bodies, np.asarray([[0, 1], [0, 1]]),
                                    np.asarray([], dtype=int), rigid_limit=1, soft_limit=1)
        with self.assertRaisesRegex(ValueError, "soft contact"):
            validate_contact_counts(shape_bodies, np.empty((0, 2), dtype=int),
                                    np.asarray([2, 2]), rigid_limit=1, soft_limit=1)


if __name__ == "__main__":
    unittest.main()
