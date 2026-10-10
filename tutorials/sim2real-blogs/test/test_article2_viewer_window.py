"""Public renderer lifecycle mocks only; no graphics, physics or process launch."""
import importlib.util
import ast
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'notebooks/mujoco/part1/viewer_window.py'
spec=importlib.util.spec_from_file_location('article2_teaching_viewer_window',PATH)
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)


class FakeAPI:
    def __init__(self):
        self.events=[]; self.callbacks={};self.closed=False;self.fail=None
        self.glfw=ModuleType('glfw');self.mj=ModuleType('mujoco')
        def record(name,result=None):
            def call(*args,**kw):
                self.events.append(name)
                if self.fail==name:raise RuntimeError(name+' failed')
                return result
            return Mock(side_effect=call)
        for n in ('PRESS','MOUSE_BUTTON_LEFT','MOUSE_BUTTON_RIGHT','MOUSE_BUTTON_MIDDLE','KEY_LEFT_SHIFT','KEY_RIGHT_SHIFT'):
            setattr(self.glfw,n,n)
        self.glfw.init=record('init',True)
        self.glfw.create_window=record('window',object())
        for n in ('make_context_current','swap_interval','swap_buffers','poll_events','destroy_window','terminate'):
            setattr(self.glfw,n,record(n))
        self.glfw.window_should_close=lambda window:self.closed
        self.glfw.get_framebuffer_size=lambda window:(800,600)
        self.glfw.get_window_size=lambda window:(800,600)
        self.glfw.get_mouse_button=lambda *args:False
        self.glfw.get_key=lambda *args:False
        for k in ('key','cursor_pos','scroll'):
            setattr(self.glfw,'set_'+k+'_callback',lambda w,fn,k=k:self.callbacks.update({k:fn}))
        self.mj.MjvCamera=lambda:SimpleNamespace()
        self.mj.MjvOption=lambda:SimpleNamespace()
        self.mj.MjvPerturb=lambda:SimpleNamespace()
        self.mj.MjvScene=record('scene',object())
        self.context=SimpleNamespace(free=record('free'))
        self.mj.MjrContext=record('context',self.context)
        self.mj.mjtFontScale=SimpleNamespace(mjFONTSCALE_150=150)
        self.mj.mjtCatBit=SimpleNamespace(mjCAT_ALL='ALL')
        self.mj.mjtMouse=SimpleNamespace(**{n:n for n in ('mjMOUSE_MOVE_H','mjMOUSE_MOVE_V','mjMOUSE_ROTATE_H','mjMOUSE_ROTATE_V','mjMOUSE_ZOOM')})
        self.mj.MjrRect=lambda *args:args
        for n in ('mjv_defaultCamera','mj_forward','mjv_updateScene','mjr_render','mjv_moveCamera'):
            setattr(self.mj,n,record(n))
    def modules(self):return patch.dict(sys.modules,{'glfw':self.glfw,'mujoco':self.mj})


class WindowTests(unittest.TestCase):
    def test_copies_identical(self):
        self.assertEqual(PATH.read_bytes(),(ROOT/'notebooks/mujoco/part2/viewer_window.py').read_bytes())

    def test_owned_context_teardown_exact_order_and_idempotent(self):
        api=FakeAPI();model=object();data=object()
        with api.modules():
            with W.launch(model,data,key_callback=Mock()) as viewer:
                viewer.sync();self.assertTrue(viewer.is_running())
            viewer.close()
        self.assertEqual(api.events[-3:],['free','destroy_window','terminate'])
        for n in ('free','destroy_window','terminate'):self.assertEqual(api.events.count(n),1)
        self.assertFalse(viewer.is_running())
        api.mj.mj_forward.assert_called_once_with(model,data)
        api.glfw.swap_interval.assert_called_once_with(0)
        api.mj.mjv_updateScene.assert_called_once()
        self.assertNotIn('mj_step',api.events)

    def test_failed_window_context_and_render_still_cleanup(self):
        for stage,expected in [('context',['destroy_window','terminate']),('mjr_render',['free','destroy_window','terminate'])]:
            api=FakeAPI();api.fail=stage
            with api.modules(),self.assertRaisesRegex(RuntimeError,stage):
                with W.launch(object(),object(),key_callback=Mock()) as viewer:viewer.sync()
            self.assertEqual(api.events[-len(expected):],expected)
        api=FakeAPI();api.glfw.create_window=Mock(return_value=None)
        with api.modules(),self.assertRaisesRegex(RuntimeError,'create'):
            with W.launch(object(),object(),key_callback=Mock()):pass
        self.assertEqual(api.events[-1],'terminate');api.glfw.destroy_window.assert_not_called()

    def test_failed_context_free_still_destroys_window_and_propagates(self):
        api=FakeAPI();api.fail='free'
        with api.modules(),self.assertRaisesRegex(RuntimeError,'free'):
            with W.launch(object(),object(),key_callback=Mock()):pass
        self.assertEqual(api.events[-3:],['free','destroy_window','terminate'])

    def test_keyboard_interrupt_and_real_error_propagate_after_cleanup(self):
        for exc in (KeyboardInterrupt(),ValueError('physics failed')):
            api=FakeAPI()
            with api.modules(),self.assertRaises(type(exc)):
                with W.launch(object(),object(),key_callback=Mock()):raise exc
            self.assertEqual(api.events[-3:],['free','destroy_window','terminate'])

    def test_only_press_delegates_key_once_and_camera_controls_are_visual(self):
        api=FakeAPI();callback=Mock()
        with api.modules():
            with W.launch(object(),object(),key_callback=callback) as viewer:
                for key in (32,256,81):
                    api.callbacks['key'](viewer.window,key,0,api.glfw.PRESS,0)
                    api.callbacks['key'](viewer.window,key,0,'RELEASE',0)
                    api.callbacks['key'](viewer.window,key,0,'REPEAT',0)
                self.assertEqual([c.args for c in callback.call_args_list],[(32,),(256,),(81,)])
                viewer._cursor(viewer.window,1,2)
                api.glfw.get_mouse_button=lambda window,key:api.glfw.PRESS if key==api.glfw.MOUSE_BUTTON_LEFT else False
                viewer._cursor(viewer.window,4,8);viewer._scroll(viewer.window,0,1)
                self.assertEqual(api.mj.mjv_moveCamera.call_count,2)
                self.assertEqual(api.mj.mj_forward.call_count,1)

    def test_minimized_and_close_pump_or_stop_without_render(self):
        api=FakeAPI()
        with api.modules():
            with W.launch(object(),object(),key_callback=Mock()) as viewer:
                api.glfw.get_framebuffer_size=lambda w:(0,0)
                viewer.sync();api.glfw.poll_events.assert_called_once();api.mj.mjr_render.assert_not_called()
                api.closed=True;self.assertFalse(viewer.is_running());viewer.sync()
                self.assertEqual(api.glfw.poll_events.call_count,1)

    def test_nonmain_thread_refused_before_any_glfw_call(self):
        api=FakeAPI()
        with api.modules(),patch.object(W.threading,'current_thread',return_value=object()),self.assertRaisesRegex(RuntimeError,'main thread'):
            with W.launch(object(),object(),key_callback=Mock()):pass
        self.assertEqual(api.events,[])

    def test_no_private_viewer_threads_or_delayed_exit(self):
        s=PATH.read_text()
        for bad in ('mujoco.viewer','_simulate','Thread(', 'sleep(', 'os._exit', 'atexit', 'set_error_callback'):
            self.assertNotIn(bad,s)

ENTRIES = (
    'notebooks/mujoco/part1/so101_pick_place.py',
    'notebooks/mujoco/part1/solutions/so101_pick_place_solution.py',
    'notebooks/mujoco/part2/so101_mjwarp.py',
    'notebooks/mujoco/part2/solutions/so101_mjwarp_solution.py',
)

class MacLaunchTests(unittest.TestCase):
    def test_python_main_but_not_os_main_is_refused_before_glfw(self):
        api = FakeAPI()
        main = Mock(return_value=0)
        with api.modules(), patch.object(W.sys, 'platform', 'darwin'), \
             patch.object(ctypes, 'CDLL', return_value=SimpleNamespace(pthread_main_np=main)) as load:
            with self.assertRaisesRegex(RuntimeError, 'ordinary python, not mjpython'):
                with W.launch(object(), object(), key_callback=Mock()):
                    self.fail('window must not open')
        load.assert_called_once_with(None)
        main.assert_called_once_with()
        self.assertEqual(main.argtypes, [])
        self.assertIs(main.restype, ctypes.c_int)
        self.assertEqual(api.events, [])


    def test_actual_os_main_may_use_same_synchronous_lifecycle(self):
        api = FakeAPI()
        main = Mock(return_value=1)
        with api.modules(), patch.object(W.sys, 'platform', 'darwin'), \
             patch.object(ctypes, 'CDLL', return_value=SimpleNamespace(pthread_main_np=main)):
            with W.launch(object(), object(), key_callback=Mock()) as viewer:
                viewer.sync()
        self.assertEqual(api.events[-3:], ['free', 'destroy_window', 'terminate'])
        api.glfw.swap_interval.assert_called_once_with(0)


    def test_non_darwin_never_resolves_or_calls_os_thread_api(self):
        for platform in ('linux', 'win32'):
            api = FakeAPI()
            with self.subTest(platform=platform), api.modules(), \
                 patch.object(W.sys, 'platform', platform), \
                 patch.object(ctypes, 'CDLL', side_effect=AssertionError('Darwin API called')):
                with W.launch(object(), object(), key_callback=Mock()) as viewer:
                    viewer.sync()
            self.assertEqual(api.events[-3:], ['free', 'destroy_window', 'terminate'])


    def test_entrypoint_no_longer_reexecs_any_platform_or_headless_path(self):
        for name in ENTRIES:
            tree = ast.parse((ROOT / name).read_text())
            entry = next(n for n in tree.body if isinstance(n, ast.If) and
                         ast.unparse(n.test) == "__name__ == '__main__'")
            for platform in ('darwin', 'linux', 'win32'):
                called = Mock()
                with self.subTest(file=name, platform=platform):
                    # Execute only the real entrypoint AST with a fake main;
                    # imports, model construction and physics are never run.
                    code = compile(ast.fix_missing_locations(ast.Module(body=[entry], type_ignores=[])), name, 'exec')
                    exec(code, {'__name__': '__main__', 'main': called})
                    called.assert_called_once_with()
            self.assertNotIn('maybe_relaunch_with_mjpython', (ROOT / name).read_text())
            self.assertNotIn('mjpython_viewer_error', (ROOT / name).read_text())


if __name__=='__main__':unittest.main(verbosity=2)
