"""Public viewer-control tests only; no Newton/Warp/physics/OpenGL imports."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=(Path(__file__).resolve().parents[1] / "notebooks")
spec=importlib.util.spec_from_file_location('newton_teaching_viewer_loop',ROOT/'newton/part3/viewer_loop.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

class Clock:
 def __init__(self):self.now=0.;self.sleeps=[];self.on_sleep=None
 def perf_counter(self):return self.now
 def sleep(self,dt):
  assert dt>0;self.sleeps.append(dt)
  if self.on_sleep:self.on_sleep()
  self.now+=dt

class Viewer:
 def __init__(self):self.running=True;self.close_count=0;self.allowed=[];self.should_step_calls=0;self.hidden=0
 def is_running(self):return self.running
 def should_step(self):
  self.should_step_calls+=1
  return self.allowed.pop(0) if self.allowed else True
 def close(self):self.close_count+=1;self.running=False
 def hide_loading_splash(self):self.hidden+=1

class Example:
 def __init__(self):
  self.steps=0;self.rendered=[];self.on_step=None;self.on_render=None
  self.post=0;self.final=0
  self.state_0='s0';self.state_1='s1';self.model='m';self.control='c';self.contacts='contacts'
 def step(self):
  if self.on_step:self.on_step()
  self.steps+=1
 def render(self):
  self.rendered.append(self.steps)
  if self.on_render:self.on_render()
 def test_post_step(self):self.post+=1
 def test_final(self):self.final+=1

class Harness:
 def __init__(self):
  self.viewer=Viewer();self.example=Example();self.clock=Clock();self.out=io.StringIO();self.finished=[];self.posts=[]
  self.factory=lambda:self.example
 def run(self,frames=3,render_fps=50,finish=None):
  with patch.object(helper,'time',self.clock),contextlib.redirect_stdout(self.out):
   return helper.run_viewer_frames(self.viewer,self.factory,frames=frames,render_fps=render_fps,
      post_step=lambda ex:self.posts.append(ex.steps),finish=finish or (lambda ex:self.finished.append(ex.steps)))

class ViewerTests(unittest.TestCase):
 def test_fixed_budget_even_if_window_never_stops(self):
  h=Harness();self.assertTrue(h.run(frames=2000,render_fps=None))
  self.assertEqual(h.example.steps,2000);self.assertEqual(len(h.example.rendered),2000)
  self.assertEqual(h.finished,[2000]);self.assertEqual(h.viewer.close_count,1)
  self.assertEqual(h.viewer.should_step_calls,2000);self.assertEqual(h.viewer.hidden,1)

 def test_pause_and_single_step_consume_only_actual_physics_frames(self):
  h=Harness();h.viewer.allowed=[False,False,True,False,True,True]
  self.assertTrue(h.run())
  self.assertEqual(h.example.rendered,[0,0,1,1,2,3]);self.assertEqual(h.posts,[1,2,3])
  self.assertEqual(h.finished,[3]);self.assertEqual(h.viewer.should_step_calls,6)
  self.assertEqual(len(h.clock.sleeps),5)

 def test_early_window_close_never_finishes(self):
  h=Harness();h.example.on_render=lambda:setattr(h.viewer,'running',False)
  self.assertFalse(h.run());self.assertEqual(h.example.steps,1);self.assertEqual(h.finished,[])
  self.assertEqual(h.viewer.close_count,1);self.assertIn('Cancelled after 1/3',h.out.getvalue())

 def test_closed_before_first_frame_never_finishes(self):
  h=Harness();h.viewer.running=False
  self.assertFalse(h.run());self.assertEqual(h.example.steps,0);self.assertEqual(h.finished,[])
  self.assertEqual(h.viewer.close_count,1)

 def test_ctrl_c_at_constructor_step_render_sleep_and_finish_closes_once(self):
  def interrupt(*a):raise KeyboardInterrupt()
  for where in ('constructor','step','render','sleep','finish'):
   with self.subTest(where=where):
    h=Harness();finish=None
    if where=='constructor':h.factory=interrupt
    elif where=='step':h.example.on_step=interrupt
    elif where=='render':h.example.on_render=interrupt
    elif where=='sleep':h.clock.on_sleep=interrupt
    else:finish=interrupt
    self.assertFalse(h.run(finish=finish));self.assertEqual(h.finished,[])
    self.assertEqual(h.viewer.close_count,1);self.assertNotIn('PASS',h.out.getvalue())

 def test_real_errors_at_constructor_step_render_and_finish_propagate(self):
  def fail(*a):raise RuntimeError('real failure')
  for where in ('constructor','step','render','finish'):
   with self.subTest(where=where):
    h=Harness();finish=None
    if where=='constructor':h.factory=fail
    elif where=='step':h.example.on_step=fail
    elif where=='render':h.example.on_render=fail
    else:finish=fail
    with self.assertRaisesRegex(RuntimeError,'real failure'):h.run(finish=finish)
    self.assertEqual(h.viewer.close_count,1)

 def test_invalid_budget_and_render_rate_close_allocated_viewer(self):
  for frames,fps in [(0,50),(True,50),(1.5,50),(3,0),(3,-1),(3,float('nan')),(3,float('inf'))]:
   h=Harness()
   with self.subTest(frames=frames,fps=fps),self.assertRaises(ValueError):h.run(frames,fps)
   self.assertEqual(h.viewer.close_count,1);self.assertEqual(h.example.steps,0)

 def test_slow_render_never_skips_steps_or_sleeps_negative(self):
  h=Harness();h.example.on_render=lambda:setattr(h.clock,'now',h.clock.now+.1)
  self.assertTrue(h.run());self.assertEqual(h.example.steps,3);self.assertEqual(h.clock.sleeps,[])

 def stack_run(self,h,args,nan=None):
  package=ModuleType('newton');package.__path__=[];examples=ModuleType('newton.examples');package.examples=examples
  seen=[]
  def find(value):seen.append(value);return ['bad'] if value==nan else []
  examples.find_nan_members=find
  with patch.dict(sys.modules,{'newton':package,'newton.examples':examples}),patch.object(helper,'time',h.clock),contextlib.redirect_stdout(h.out):
   result=helper.run_stack_gl(h.viewer,h.factory,args)
  return result,seen

 def test_legacy_test_checks_preserved(self):
  h=Harness();result,seen=self.stack_run(h,SimpleNamespace(test=True,num_frames=3,render_fps=None))
  self.assertTrue(result);self.assertEqual(h.example.post,3);self.assertEqual(h.example.final,1)
  self.assertEqual(seen,['s0','s1','m','c','contacts']);self.assertEqual(h.viewer.close_count,1)

 def test_legacy_nan_validation_failure_is_not_hidden(self):
  h=Harness()
  with self.assertRaisesRegex(ValueError,'NaN members found in model'):
   self.stack_run(h,SimpleNamespace(test=True,num_frames=3,render_fps=None),nan='m')
  self.assertEqual(h.viewer.close_count,1)

 def test_legacy_without_test_does_not_claim_physical_success(self):
  h=Harness();result,seen=self.stack_run(h,SimpleNamespace(test=False,num_frames=3,render_fps=None))
  self.assertTrue(result);self.assertEqual(seen,[]);self.assertEqual(h.example.final,0)
  self.assertIn('use --test',h.out.getvalue());self.assertNotIn('PASS',h.out.getvalue())

 def test_legacy_invalid_zero_rate_not_silently_defaulted(self):
  h=Harness()
  with self.assertRaises(ValueError):self.stack_run(h,SimpleNamespace(test=True,num_frames=3,render_fps=0))
  self.assertEqual(h.viewer.close_count,1)

if __name__=='__main__':unittest.main()
