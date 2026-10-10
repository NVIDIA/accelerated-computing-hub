"""UI lifecycle tests with fake viewers and clocks; no physics or OpenGL."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/mujoco/part1/viewer_loop.py'
spec=importlib.util.spec_from_file_location('article2_teaching_viewer_loop',path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

class Clock:
    def __init__(self):self.now=0.;self.sleeps=[];self.on_sleep=None
    def perf_counter(self):return self.now
    def sleep(self,seconds):
        assert seconds>0
        self.sleeps.append(seconds)
        if self.on_sleep:self.on_sleep()
        self.now+=seconds

class Viewer:
    def __init__(self):
        self.running=True;self.enters=0;self.exits=0;self.syncs=0;self.callback=None;self.on_sync=None
        self.lock_depth=0;self.lock_entries=0
    def __enter__(self):self.enters+=1;return self
    def __exit__(self,*exc):self.exits+=1;self.running=False
    def is_running(self):return self.running
    def sync(self):
        self.syncs+=1
        if self.on_sync:self.on_sync()
    def close(self):raise AssertionError('Helper must use one context exit, not a second close')
    @contextlib.contextmanager
    def lock(self):
        self.lock_entries+=1;self.lock_depth+=1
        try:yield
        finally:self.lock_depth-=1

class Harness:
    def __init__(self):
        self.viewer=Viewer();self.clock=Clock();self.frames=0;self.launches=0
        self.on_step=None;self.output=io.StringIO()
        self.package=ModuleType('mujoco');self.package.__path__=[]
        self.module=ModuleType('viewer_window');self.module.launch=self.launch
        self.package.viewer=self.module
    def launch(self,model,data,*,key_callback):
        self.launches+=1;self.viewer.callback=key_callback;return self.viewer
    def step(self):
        if self.on_step:self.on_step()
        self.frames+=1
    def run(self,frames=3,fps=50,configure_viewer=None):
        with patch.dict(sys.modules,{'viewer_window':self.module}),patch.object(helper,'time',self.clock),contextlib.redirect_stdout(self.output):
            return helper.run_passive_frames(object(),object(),self.step,frames=frames,fps=fps,configure_viewer=configure_viewer)

class ViewerLoopTests(unittest.TestCase):
    def test_identical_helper_copies(self):
        self.assertEqual(path.read_bytes(),(ROOT/'notebooks/mujoco/part2/viewer_loop.py').read_bytes())

    def test_auto_end_exact_budget_one_context_exit(self):
        h=Harness();configured=[]
        def configure(viewer):
            self.assertEqual(viewer.lock_depth,1)
            configured.append(viewer)
        self.assertTrue(h.run(configure_viewer=configure))
        self.assertEqual(h.frames,3);self.assertEqual(h.viewer.syncs,3)
        self.assertEqual(h.viewer.enters,h.viewer.exits);self.assertEqual(h.viewer.exits,1)
        self.assertEqual(configured,[h.viewer]);self.assertEqual(len(h.clock.sleeps),2)
        self.assertEqual(h.viewer.lock_entries,1);self.assertEqual(h.viewer.lock_depth,0)
        self.assertNotIn('PASS',h.output.getvalue());self.assertNotIn('Cancelled',h.output.getvalue())

    def test_pause_pumps_events_without_consuming_physics_budget(self):
        h=Harness();observed=[]
        def configure(viewer):viewer.callback(32)
        def sync():
            observed.append(h.frames)
            if h.viewer.syncs==2:h.viewer.callback(32)
        h.viewer.on_sync=sync
        self.assertTrue(h.run(configure_viewer=configure))
        self.assertEqual(observed,[0,0,1,2,3]);self.assertEqual(h.frames,3)
        self.assertEqual(h.viewer.exits,1);self.assertEqual(len(h.clock.sleeps),4)

    def test_escape_and_q_cancel_before_any_frame(self):
        for key in (256,ord('q'),ord('Q')):
            with self.subTest(key=key):
                h=Harness()
                self.assertFalse(h.run(configure_viewer=lambda viewer:viewer.callback(key)))
                self.assertEqual(h.frames,0);self.assertEqual(h.viewer.exits,1)
                self.assertIn('Cancelled after 0/3',h.output.getvalue());self.assertNotIn('PASS',h.output.getvalue())

    def test_escape_after_sync_does_not_step_again(self):
        h=Harness();h.viewer.on_sync=lambda:h.viewer.callback(256)
        self.assertFalse(h.run());self.assertEqual(h.frames,1);self.assertEqual(h.viewer.syncs,1)
        self.assertEqual(h.clock.sleeps,[]);self.assertEqual(h.viewer.exits,1)

    def test_window_close_reports_incomplete(self):
        h=Harness();h.viewer.on_sync=lambda:setattr(h.viewer,'running',False)
        self.assertFalse(h.run());self.assertEqual(h.frames,1);self.assertEqual(h.viewer.exits,1)
        self.assertIn('not checked',h.output.getvalue())

    def test_interrupt_in_step_sleep_and_sync_closes_once_without_traceback(self):
        def interrupt():raise KeyboardInterrupt()
        for phase,expected in [('step',0),('sleep',1),('sync',1)]:
            with self.subTest(phase=phase):
                h=Harness()
                if phase=='step':h.on_step=interrupt
                elif phase=='sleep':h.clock.on_sleep=interrupt
                else:h.viewer.on_sync=interrupt
                self.assertFalse(h.run());self.assertEqual(h.frames,expected)
                self.assertEqual(h.viewer.exits,1);self.assertNotIn('Traceback',h.output.getvalue())
                self.assertNotIn('PASS',h.output.getvalue())

    def test_physics_error_propagates_after_one_cleanup(self):
        h=Harness()
        def error():raise RuntimeError('real physics failure')
        h.on_step=error
        with self.assertRaisesRegex(RuntimeError,'real physics failure'):h.run()
        self.assertEqual(h.viewer.exits,1);self.assertNotIn('PASS',h.output.getvalue())

    def test_configuration_error_releases_lock_and_closes_once(self):
        h=Harness()
        def configure(viewer):
            self.assertEqual(viewer.lock_depth,1)
            raise RuntimeError('camera setup failure')
        with self.assertRaisesRegex(RuntimeError,'camera setup failure'):h.run(configure_viewer=configure)
        self.assertEqual(h.viewer.lock_depth,0);self.assertEqual(h.viewer.exits,1)
        self.assertEqual(h.frames,0)

    def test_simulation_and_sync_run_outside_camera_lock(self):
        h=Harness()
        h.on_step=lambda:self.assertEqual(h.viewer.lock_depth,0)
        h.viewer.on_sync=lambda:self.assertEqual(h.viewer.lock_depth,0)
        self.assertTrue(h.run(configure_viewer=lambda viewer:None))
        self.assertEqual(h.viewer.lock_entries,1)

    def test_invalid_budget_rejected_before_window_launch(self):
        for frames,fps in [(0,50),(-1,50),(1.5,50),(True,50),(3,0),(3,-1),(3,float('nan')),(3,float('inf')),(3,True)]:
            with self.subTest(frames=frames,fps=fps):
                h=Harness()
                with self.assertRaises(ValueError):h.run(frames,fps)
                self.assertEqual(h.launches,0)

    def test_slow_frame_does_not_sleep_or_skip_physics(self):
        h=Harness();h.on_step=lambda:setattr(h.clock,'now',h.clock.now+.1)
        self.assertTrue(h.run());self.assertEqual(h.frames,3);self.assertEqual(h.clock.sleeps,[])

if __name__=='__main__':unittest.main()
