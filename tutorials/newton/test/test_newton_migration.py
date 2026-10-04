"""Real Newton 1.6.0 regression tests; use the dedicated Article 3 environment.

    newton/.venv/bin/python -m unittest discover -s tests -p test_newton_migration.py -v

Robot probes run in separate processes: the shared IK helper has robot-global
layout aliases. These are CPU/native-MuJoCo checks, not CUDA verification.
"""
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = (Path(__file__).resolve().parents[1] / "notebooks")
PART3 = ROOT / "newton" / "part3"


class NewtonMigrationTests(unittest.TestCase):
    def test_dedicated_requirements_pin_the_exercised_physics_stack(self):
        import importlib.metadata

        path = ROOT / 'newton' / 'requirements.txt'
        self.assertTrue(path.is_file(), 'Article 3 needs its own requirements, not an Article 2 downgrade')
        requirements = {line.strip() for line in path.read_text().splitlines() if not line.startswith('#')}
        for package, version in (('newton[examples]', '1.6.0'), ('warp-lang', '1.17.0'),
                                 ('mujoco', '3.12.0'), ('mujoco-warp', '3.12.0'),
                                 ('newton-usd-schemas', '0.5.0'), ('usd-core', '26.3')):
            self.assertIn(f'{package}=={version}', requirements)
            self.assertEqual(importlib.metadata.version(package.split('[')[0]), version)
        for package in ('nbformat', 'nbclient', 'ipykernel', 'pytest'):
            self.assertTrue(any(line.startswith(package + '=') or line.startswith(package + '>')
                                for line in requirements), package)

    def run_probe(self, robot: str, code: str, *, timeout: int = 420) -> str:
        preamble = f"""
import faulthandler
faulthandler.dump_traceback_later(120, repeat=True)
import sys
sys.path.insert(0, {str(PART3)!r})
sys.path.insert(0, {str(PART3 / 'solutions')!r})
import numpy as np
import warp as wp
import newton
wp.set_device('cpu')
from robots import get_robot
from newton_scene_solution import build_newton_model
spec = get_robot({robot!r})
"""
        result = subprocess.run(
            [sys.executable, "-u", "-c", preamble + textwrap.dedent(code)],
            cwd=ROOT, text=True, capture_output=True, timeout=timeout,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_both_robot_models_construct_with_coordinate_targets(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                self.run_probe(robot, """
                    model = build_newton_model(spec)
                    assert newton.use_coord_layout_targets is True
                    assert model.joint_target_q.shape == model.joint_q.shape
                    assert np.isfinite(model.joint_q.numpy()).all()
                    assert np.isfinite(model.body_q.numpy()).all()
                """)

    def test_cpu_frame_uses_native_contacts_and_reference_timestep(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                self.run_probe(robot, """
                    from newton.viewer import ViewerNull
                    from so101_newton_solution import Example
                    args = Example.create_parser().parse_args(['--robot', spec.key])
                    example = Example(ViewerNull(num_frames=1), args)
                    assert abs(example.sim_dt - 0.002) < 1e-12, example.sim_dt
                    assert not example.use_newton_contacts, 'Native MuJoCo ignores Newton Contacts'
                    assert example.contacts is None, 'Do not compute unused Newton contacts on native CPU'
                    assert example.graph is None
                    example.step()
                    example.test_post_step()
                    np.testing.assert_allclose(example.sim_time, 0.02)
                    assert np.isfinite(example.state_0.body_q.numpy()).all()
                    assert np.isfinite(example.state_0.body_qd.numpy()).all()
                """)

    def test_solver_keeps_mjcf_gripper_limits_and_applies_only_arm_boost(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                self.run_probe(robot, """
                    import pick_place_common as task
                    from newton.viewer import ViewerNull
                    from so101_newton_solution import Example
                    from newton_scene_solution import robot_joint_indices
                    args = Example.create_parser().parse_args(['--robot', spec.key])
                    example = Example(ViewerNull(num_frames=1), args)
                    scene = task.resolve_pick_place_scene(spec=spec)
                    reference = task.load_pick_place_model(scene, spec)
                    mj = example.solver.mj_model
                    np.testing.assert_allclose(mj.actuator_forcerange, reference.actuator_forcerange, rtol=1e-6)
                    np.testing.assert_array_equal(mj.actuator_forcelimited, reference.actuator_forcelimited)
                    joints = robot_joint_indices(example.model, reference)
                    dofs = example.model.joint_qd_start.numpy()[joints]
                    np.testing.assert_allclose(example.model.joint_target_ke.numpy()[dofs], spec.arm_target_ke)
                    np.testing.assert_allclose(example.model.joint_target_kd.numpy()[dofs], spec.arm_target_kd)
                    np.testing.assert_allclose(example.model.joint_q.numpy()[example.target_indices], spec.home_ctrl)
                """)

    def test_contact_gap_matches_zero_gap_mjcf_instead_of_newton_default(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                self.run_probe(robot, """
                    from newton.viewer import ViewerNull
                    from so101_newton_solution import Example
                    args = Example.create_parser().parse_args(['--robot', spec.key])
                    example = Example(ViewerNull(num_frames=1), args)
                    np.testing.assert_array_equal(example.ik_model.geom_gap, 0.0)
                    # Sites have no physical contacts and keep their own defaults.
                    colliders = (example.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
                    np.testing.assert_array_equal(example.model.shape_gap.numpy()[colliders], 0.0)
                    np.testing.assert_array_equal(example.solver.mj_model.geom_gap, 0.0)
                """)

    def test_newton_contacts_refresh_on_each_real_mujoco_warp_substep(self):
        self.run_probe('so101', """
            import mujoco_warp as mjw
            from so101_newton_solution import Example
            from so101_newton import Example as ExerciseExample
            newton.use_coord_layout_targets = True
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.gap = 0.0
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.025)
            builder.add_ground_plane()
            # Exercise the real scheduling method with a tiny model, not meshes.
            # MuJoCo Warp runs on CPU here; this is NOT a CUDA validation.
            example = Example.__new__(Example)
            example.model = builder.finalize()
            example.state_0 = example.model.state()
            example.state_1 = example.model.state()
            example.control = example.model.control()
            example.collision_pipeline = newton.CollisionPipeline(example.model)
            example.contacts = example.collision_pipeline.contacts()
            example.contact_peak = wp.zeros(1, dtype=wp.int32, device=example.model.device)
            example.use_newton_contacts = True
            example.sim_substeps = 3
            example.sim_dt = 0.002
            newton.eval_fk(example.model, example.model.joint_q, example.model.joint_qd, example.state_0)
            example.solver = newton.solvers.SolverMuJoCo(
                example.model, use_mujoco_cpu=False, use_mujoco_contacts=False,
                njmax=32, nconmax=16, integrator='implicitfast',
            )
            detections = []
            collide = example.collision_pipeline.collide
            def observe_real_collision(state, contacts):
                collide(state, contacts)
                detections.append((state.body_q.numpy().copy(), int(contacts.rigid_contact_count.numpy()[0])))
            example.collision_pipeline.collide = observe_real_collision
            example.simulate()
            assert len(detections) == example.sim_substeps, len(detections)
            assert any(count > 0 for _, count in detections)
            assert not np.array_equal(detections[0][0], detections[-1][0])
            for _ in range(50):
                example.simulate()
            assert np.isfinite(example.state_0.body_q.numpy()).all()
            assert abs(float(example.state_0.body_q.numpy()[body, 2]) - 0.025) < 0.005
            for check in (Example.test_post_step, ExerciseExample.test_post_step):
                check(example)

            # An actual undersized MJWarp constraint buffer must fail the
            # checker, even when the body state remains finite. Newton's
            # own contact detector generates more than one constraint row.
            example.solver = newton.solvers.SolverMuJoCo(
                example.model, use_mujoco_cpu=False, use_mujoco_contacts=False,
                njmax=1, nconmax=16, integrator='implicitfast',
            )
            newton.eval_fk(example.model, example.model.joint_q, example.model.joint_qd, example.state_0)
            example.simulate()
            assert (example.solver.mjw_data.overflow.numpy() & mjw.OverflowType.NEFC).any()
            for check in (Example.test_post_step, ExerciseExample.test_post_step):
                try:
                    check(example)
                except ValueError as error:
                    assert 'capacity overflow' in str(error), error
                else:
                    raise AssertionError('Real MJWarp constraint overflow was accepted')
        """)

    def test_newton_contact_peak_rejects_both_truncation_boundaries(self):
        import json
        import re

        notebook_path = PART3 / '02__mujoco_to_newton.ipynb'
        notebook = json.loads(notebook_path.read_text())
        step_eight, = (
            ''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'markdown'
            and ''.join(cell['source']).startswith('### Step 8.')
        )
        notebook_loop, = (
            code for code in re.findall(r'```python\n(.*?)```', step_eight, re.S)
            if code.startswith('for _ in range(self.sim_substeps):')
        )
        self.run_probe('so101', f"""
            from pathlib import Path
            import so101_newton as exercise
            from so101_newton_solution import Example
            newton.use_coord_layout_targets = True
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.gap = 0.0
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            for x in (-0.1, 0.1):
                body = builder.add_body(xform=wp.transform(wp.vec3(x, 0, 0.02), wp.quat_identity()))
                builder.add_shape_sphere(body, radius=0.025)
            builder.add_ground_plane()
            model = builder.finalize()
            snippet = Path({str(PART3 / 'solutions' / 'step_08_substep_loop.py')!r})
            snippet_code = compile(snippet.read_text(), str(snippet), 'exec')
            notebook_code = compile({notebook_loop!r}, {str(notebook_path)!r} + ':Step8', 'exec')
            # Real contacts overflow either Newton's own allocation or the
            # smaller Newton-to-MJWarp transfer buffer. The latter clips before
            # filtering contacts and does not set MJWarp's overflow flags.
            # Execute the literal teaching loop at both boundaries as well.
            for loop_source, pipeline_capacity, transfer_capacity in (
                ('reference', 16, 1), ('solution_snippet', 1, 16),
                ('notebook', 16, 1), ('notebook', 1, 16),
            ):
                example = Example.__new__(Example)
                example.model = model
                example.state_0, example.state_1 = model.state(), model.state()
                example.control = model.control()
                example.collision_pipeline = newton.CollisionPipeline(model, rigid_contact_max=pipeline_capacity)
                example.contacts = example.collision_pipeline.contacts()
                example.contact_peak = wp.zeros(1, dtype=wp.int32, device=model.device)
                example.sim_substeps, example.sim_dt = 1, 0.002
                example.solver = newton.solvers.SolverMuJoCo(
                    model, use_mujoco_cpu=False, use_mujoco_contacts=False,
                    njmax=32, nconmax=transfer_capacity, integrator='implicitfast',
                )
                newton.eval_fk(model, model.joint_q, model.joint_qd, example.state_0)
                # Markdown teaching code must preserve the real contact peak,
                # just like the full reference and the external step solution.
                def advance():
                    if loop_source == 'reference':
                        example.simulate()
                    else:
                        code = snippet_code if loop_source == 'solution_snippet' else notebook_code
                        exec(code, dict(self=example, wp=wp,
                                        _record_contact_peak=exercise._record_contact_peak))
                advance()
                assert int(example.contact_peak.numpy()[0]) == 2, (loop_source, pipeline_capacity, transfer_capacity)
                # Move both bodies clear for a subsequent real collision pass.
                # Checking only this last contact count would miss the loss.
                q = example.state_0.body_q.numpy()
                q[:, 2] = 1.0
                example.state_0.body_q.assign(q)
                advance()
                assert int(example.contacts.rigid_contact_count.numpy()[0]) == 0
                assert int(example.contact_peak.numpy()[0]) == 2, (loop_source, pipeline_capacity, transfer_capacity)
                for check in (Example.test_post_step, exercise.Example.test_post_step):
                    try:
                        check(example)
                    except ValueError as error:
                        assert 'Newton contact capacity overflow' in str(error), error
                    else:
                        raise AssertionError('A previous Newton contact truncation was accepted')
        """)

    def test_both_robots_physically_stack_after_600_frames(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                output = self.run_probe(robot, """
                    from newton.viewer import ViewerNull
                    from so101_newton_solution import Example
                    args = Example.create_parser().parse_args(['--robot', spec.key])
                    example = Example(ViewerNull(num_frames=600), args)
                    for frame in range(600):
                        example.step()
                        assert np.isfinite(example.state_0.body_q.numpy()).all(), frame
                        assert np.isfinite(example.state_0.body_qd.numpy()).all(), frame
                    q = example.state_0.body_q.numpy()
                    red, blue = q[example.red_body, :3], q[example.blue_body, :3]
                    xy = float(np.linalg.norm(red[:2] - blue[:2]))
                    dz = float(red[2] - blue[2])
                    print(f'PHYSICAL {spec.key}: frames=600 xy={xy:.8f} dz={dz:.8f}')
                    assert xy <= 0.015 and 0.035 <= dz <= 0.055, (red, blue, xy, dz)
                    example.test_final()
                """)
                print('\n' + '\n'.join(line for line in output.splitlines()
                                       if line.startswith(('Physics:', 'PHYSICAL', 'red=', 'Part 3'))))

    def test_final_rejects_loose_alignment_nonfinite_and_incomplete_runs(self):
        self.run_probe('so101', """
            from newton.viewer import ViewerNull
            from so101_newton_solution import Example
            from so101_newton import Example as ExerciseExample
            example = Example(ViewerNull(num_frames=1), Example.create_parser().parse_args([]))
            base = example.state_0.body_q.numpy().copy()
            base[example.red_body, :3] = base[example.blue_body, :3] + [0, 0, 0.044]
            accepted_invalid = []
            # Deliberately manufactured bad checker inputs, not simulation results.
            for case in ('loose_xy', 'nan_position', 'nan_velocity', 'unfinished'):
                q = base.copy()
                qd = example.state_0.body_qd.numpy() * 0
                example.controller.phase = len(example.controller.sequence)
                if case == 'loose_xy':
                    q[example.red_body, 0] += 0.04
                elif case == 'nan_position':
                    q[example.red_body, 0] = np.nan
                elif case == 'nan_velocity':
                    qd[example.red_body, 0] = np.nan
                else:
                    example.controller.phase = 0
                example.state_0.body_q.assign(q)
                example.state_0.body_qd.assign(qd)
                for check in (Example.test_final, ExerciseExample.test_final):
                    try:
                        check(example)
                    except ValueError:
                        pass
                    else:
                        accepted_invalid.append((check.__module__, case))
            assert not accepted_invalid, accepted_invalid
        """)

    def test_step_snippets_build_and_step_the_same_models_as_the_solution(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                self.run_probe(robot, f"""
                    from pathlib import Path
                    from types import SimpleNamespace
                    import newton_scene as exercise
                    import pick_place_common as task
                    scene = task.resolve_pick_place_scene(spec=spec)
                    snippets = Path({str(PART3 / 'solutions')!r})
                    context = dict(vars(exercise), spec=spec, robot_xml=scene.parent / spec.robot_xml)
                    for step in range(6):
                        path, = snippets.glob(f'step_{{step:02d}}_*.py')
                        exec(compile(path.read_text(), str(path), 'exec'), context)
                    model = context['model']
                    reference = build_newton_model(spec)
                    for attr in ('joint_q', 'joint_target_q', 'joint_target_ke', 'joint_target_kd',
                                 'joint_effort_limit', 'body_mass', 'body_inertia', 'shape_gap'):
                        np.testing.assert_array_equal(getattr(model, attr).numpy(), getattr(reference, attr).numpy())
                    example = SimpleNamespace(model=model, spec=spec, sim_substeps=10, sim_dt=0.002)
                    context.update(self=example, use_mujoco_cpu=True, use_mujoco_contacts=True)
                    for step in (6, 7):
                        path, = snippets.glob(f'step_{{step:02d}}_*.py')
                        exec(compile(path.read_text(), str(path), 'exec'), context)
                    example.collision_pipeline = None
                    example.contacts = None
                    for step in (8, 9):
                        path, = snippets.glob(f'step_{{step:02d}}_*.py')
                        exec(compile(path.read_text(), str(path), 'exec'), context)
                    assert example.graph is None
                    assert np.isfinite(example.state_0.body_q.numpy()).all()
                    assert example.solver.mj_model.opt.timestep == 0.002
                    try:
                        exercise.build_newton_model(spec)
                    except NotImplementedError:
                        pass
                    else:
                        raise AssertionError('Exercise must remain student-owned and incomplete')
                """)

    def test_cube_mass_and_inertia_match_the_reference_mjcf(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                self.run_probe(robot, """
                    from newton_scene_solution import body_index, build_ik_model
                    model = build_newton_model(spec)
                    ik, _, _ = build_ik_model(spec)
                    for label in ('red_cube', 'blue_cube'):
                        body = body_index(model, label)
                        reference = ik.body(label).id
                        np.testing.assert_allclose(model.body_mass.numpy()[body], 0.08, rtol=1e-6)
                        np.testing.assert_allclose(model.body_mass.numpy()[body],
                                                   ik.body_mass[reference], rtol=1e-6)
                        np.testing.assert_allclose(model.body_inertia.numpy()[body],
                                                   np.diag(ik.body_inertia[reference]), rtol=1e-6)
                        np.testing.assert_allclose(model.body_q.numpy()[body, :3],
                                                   ik.body_pos[reference], atol=1e-7)
                """)

    def test_targets_follow_named_joints_not_dof_or_prefix_offsets(self):
        for robot in ("so101", "rebot"):
            with self.subTest(robot=robot):
                self.run_probe(robot, """
                    from newton_scene_solution import (
                        build_ik_model, robot_joint_indices, set_joint_targets,
                    )
                    ik, _, scene = build_ik_model(spec)
                    newton.use_coord_layout_targets = True
                    builder = newton.ModelBuilder()
                    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
                    # A free body before the arm separates q and qd layouts.
                    body = builder.add_body(label='uncontrolled')
                    builder.add_shape_sphere(body, radius=0.02)
                    builder.add_mjcf(str(scene.parent / spec.robot_xml),
                                     ignore_names=['floor'], collapse_fixed_joints=True)
                    model = builder.finalize()
                    joints = robot_joint_indices(model, ik)
                    indices = model.joint_target_q_start.numpy()[joints]
                    assert len(indices) == len(spec.seed_pose)
                    assert (indices != model.joint_qd_start.numpy()[joints]).all()
                    for actuator, joint in enumerate(joints):
                        name = ik.joint(int(ik.actuator_trnid[actuator, 0])).name
                        assert model.joint_label[joint].endswith('/' + name)
                    control = model.control()
                    before = control.joint_target_q.numpy().copy()
                    set_joint_targets(control, spec.home_ctrl, indices)
                    after = control.joint_target_q.numpy()
                    np.testing.assert_allclose(after[indices], spec.home_ctrl)
                    mask = np.ones(len(before), dtype=bool)
                    mask[indices] = False
                    np.testing.assert_array_equal(after[mask], before[mask])
                    # Reject malformed controller output instead of broadcasting it.
                    try:
                        set_joint_targets(control, spec.home_ctrl[:-1], indices)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('Wrong-sized target vector was accepted')
                """)


if __name__ == "__main__":
    unittest.main()
