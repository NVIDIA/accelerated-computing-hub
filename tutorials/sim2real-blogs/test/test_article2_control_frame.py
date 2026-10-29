"""Check the GPU lesson's control/physics transfer boundary without a GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "notebooks/mujoco/part2/solutions/so101_mjwarp_solution.py"


class ControlFrameTests(unittest.TestCase):
    def test_controls_upload_once_and_observations_follow_all_substeps(self):
        for box in (False, True):
            with self.subTest(box_task=box):
                events = []
                def record(name):
                    def call(*args, **kwargs):
                        events.append(name)
                    return call
                class DeviceArray:
                    def numpy(self):
                        events.append("download")
                        return np.zeros((1, 2))
                tree = ast.parse(SCRIPT.read_text())
                demo = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_demo")
                frame = next(n for n in demo.body if isinstance(n, ast.FunctionDef) and n.name == "simulate_frame")
                frame.body[0] = ast.Global(names=["frame_idx"])
                task = SimpleNamespace(observe=record("observe")) if box else None
                scope = dict(mujoco=SimpleNamespace(mj_forward=record("host-forward")), frame_idx=0, box_task=task, pending_phase="close", pending_ctrl=np.zeros(2),
                             controller=SimpleNamespace(phase_name=lambda: "close", step=lambda *args: np.zeros(2)),
                             np=np, mjm=SimpleNamespace(nu=2), mjd=SimpleNamespace(ctrl=np.zeros(2), qpos=np.zeros(2), qvel=np.zeros(2)),
                             frame_dt=.02, sim_substeps=20, fps=50, device="fake", m=object(),
                             d=SimpleNamespace(ctrl=object(), qpos=DeviceArray(), qvel=DeviceArray()),
                             wp=SimpleNamespace(copy=record("upload"), array=lambda *args, **kwargs: object(), float32="float32"),
                             mjw=SimpleNamespace(step=record("step"), forward=record("forward"), get_data_into=record("download-all")),
                             assert_mjwarp_capacity=record("capacity"))
                exec(compile(ast.fix_missing_locations(ast.Module(body=[frame], type_ignores=[])), str(SCRIPT), "exec"), scope)
                scope["simulate_frame"]()
                self.assertEqual(events.count("upload"), 1)
                self.assertEqual(events.count("step"), 20)
                self.assertLess(events.index("upload"), events.index("step"))
                last_step = max(i for i, event in enumerate(events) if event == "step")
                downloads = [i for i, event in enumerate(events) if event.startswith("download")]
                self.assertTrue(all(i > last_step for i in downloads))
                self.assertEqual(len(downloads), 1 if box else 2)
                if box:
                    self.assertEqual(events[-1], "observe")
                    self.assertEqual(events.count("capacity"), 21)
