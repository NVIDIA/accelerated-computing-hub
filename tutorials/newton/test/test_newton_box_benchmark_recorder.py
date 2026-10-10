# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Compare device observation math with native MuJoCo on real contact states.

Warp CPU executes the same recording kernels used by CUDA. These tests verify
recording math and world attribution; they do not validate GPU physics or speed.
"""
from pathlib import Path
import os
import subprocess
import sys
import unittest


PART = (Path(__file__).resolve().parents[1] / "notebooks") / 'newton' / 'part3'
SCRIPT = r'''
from types import SimpleNamespace
import numpy as np
import mujoco
import warp as wp
import benchmark_box_protocol as protocol
import pick_place_common as task
from box_task import BoxTask, native_contacts
from robots import get_robot

wp.init()
wp.set_device('cpu')
for robot in ('so101', 'rebot'):
    box = BoxTask(get_robot(robot))
    scene = box.resolve_scene()
    model = task.load_pick_place_model(scene, box.spec)
    model.opt.timestep = protocol.DT
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    model.opt.impratio = 100
    native = protocol.NativeRecorder(model, get_robot(robot))
    snapshots, forces, pairs, worlds = [], [], [], []
    expected = np.zeros((2, 61), dtype=np.float32)
    for world in range(2):
        data = mujoco.MjData(model)
        controller = box.make_controller()
        first = controller.step(model, data, .02)
        task.apply_arm_ctrl(model, data, first)
        task.reset_cubes(model, data, box.spec)
        for frame in range(275 + world * 25):
            data.ctrl[:] = first if frame == 0 else controller.step(model, data, .02)
            for _ in range(protocol.SUBSTEPS):
                mujoco.mj_step(model, data)
        # Two actual grasp/lift snapshots exercise nonzero solved jaw forces.
        # Small angular velocities test COM-to-corner velocity conversion;
        # no hypothetical pose or contact force is supplied to the recorder.
        for cube, name in enumerate(('red_cube', 'blue_cube')):
            joint = model.joint(name + '_joint').id
            v = model.jnt_dofadr[joint]
            data.qvel[v:v + 6] += np.array([.01, -.03, .02, .2, -.4, .3]) * (world + 1)
        data.time = 40.
        mujoco.mj_forward(model, data)
        native.capture(data, expected[world], protocol.FRAMES - 1)
        snapshots.append(data)
        for i in range(data.ncon):
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, i, force)
            pairs.append(data.contact[i].geom.copy())
            forces.append(force)
            worlds.append(world)
    assert len(pairs) > 0, 'Fixture must contain real solved contacts'
    assert np.all(expected[:, 28:30] > 0), 'Both jaws must contact the red cube in both worlds'
    assert np.all(expected[:, 30:32] > 0), 'Both jaws must have solved normal forces'
    rec = protocol.CUDARecorder(model, None,
        SimpleNamespace(nworld=2, naconmax=len(pairs)), get_robot(robot))
    array = lambda name, dtype: wp.array(np.stack([getattr(d, name) for d in snapshots]), dtype=dtype)
    qpos = array('qpos', float)
    geom_pos = array('geom_xpos', wp.vec3)
    geom_mat = wp.array(np.stack([d.geom_xmat.reshape(-1, 3, 3) for d in snapshots]), dtype=wp.mat33)
    cvel = array('cvel', wp.spatial_vector)
    subtree_com = array('subtree_com', wp.vec3)
    sim_time = wp.array([d.time for d in snapshots], dtype=float)
    cursor = wp.array([protocol.FRAMES - 1], dtype=int)
    wp.launch(protocol._geometry, dim=(2, 2), inputs=[qpos, geom_pos, geom_mat,
        cvel, subtree_com, sim_time, rec.cubes, rec.bodies, rec.roots,
        rec.sizes, rec.signs, rec.grip_qpos, box.spec.gripper_closed,
        box.spec.gripper_open, cursor, rec.output])
    wp.launch(protocol._contact_observations, dim=len(pairs), inputs=[
        wp.array([len(pairs)], dtype=int), wp.array(pairs, dtype=wp.vec2i),
        wp.array(worlds, dtype=int), wp.array(forces, dtype=wp.spatial_vector),
        rec.cubes, rec.jaw_lookup, cursor, rec.output])
    wp.launch(protocol._jaw_clearance, dim=2, inputs=[geom_pos, geom_mat,
        rec.jaws, rec.jaw_types, rec.jaw_sizes, rec.mesh_start, rec.mesh_count,
        rec.vertices, float(max(box.spec.table_top_z, box.box.upper[2])),
        cursor, rec.output])
    actual = rec.output.numpy()[:, -1]
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6,
        err_msg=robot + ': device recorder disagrees with native observation')
    assert not np.allclose(actual[0, 3:27], actual[1, 3:27])
    print(robot, 'native/device observation agreement passed')
'''


class BoxRecorderTests(unittest.TestCase):
    def test_device_kernels_match_native_geometry_velocities_and_contacts(self):
        # A separate process isolates the article's top-level helper imports
        # from the other article when the full repository suite runs.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='box-recorder-check-') as tmp:
            program = Path(tmp) / 'compare_recorders.py'
            program.write_text(SCRIPT)
            env = dict(os.environ, PYTHONPATH=str(PART), WARP_CACHE_PATH=str(Path(tmp)/'warp'))
            result = subprocess.run([sys.executable, str(program)], env=env,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('so101 native/device observation agreement passed', result.stdout)
            self.assertIn('rebot native/device observation agreement passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
