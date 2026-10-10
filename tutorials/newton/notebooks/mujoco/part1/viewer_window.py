"""Synchronous teaching window using public GLFW and MuJoCo rendering APIs.

The calling main thread owns the window, OpenGL context, event dispatch and
teardown. Physics remains entirely in the caller's simulate_frame callback.
"""
from __future__ import annotations

from contextlib import nullcontext
import threading
import sys


class TeachingViewer:
    def __init__(self, model, data, *, key_callback):
        self.model, self.data = model, data
        self.key_callback = key_callback
        self.window = self.context = self.scene = None
        self.initialized = False
        self.closed = False
        self.last_cursor = None

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("The teaching viewer must run on the main thread")
        if sys.platform == "darwin":
            # mjpython's Python main thread is not Cocoa's OS main thread.
            # Check the public Darwin API before making any window-system call.
            import ctypes
            is_main = ctypes.CDLL(None).pthread_main_np
            is_main.argtypes = []
            is_main.restype = ctypes.c_int
            if not is_main():
                raise RuntimeError(
                    "On macOS, launch this synchronous teaching viewer with "
                    "ordinary python, not mjpython, on the OS main thread."
                )
        # Lazy imports preserve the existing headless dependency path.
        import glfw
        import mujoco
        self.glfw, self.mujoco = glfw, mujoco
        try:
            if not glfw.init():
                raise RuntimeError("GLFW could not initialize a display")
            self.initialized = True
            self.window = glfw.create_window(1280, 720, "MuJoCo teaching example", None, None)
            if not self.window:
                raise RuntimeError("GLFW could not create the teaching window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()
            self.pert = mujoco.MjvPerturb()
            mujoco.mjv_defaultCamera(self.cam)
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            # launch_passive performed this same initial forward before the loop.
            mujoco.mj_forward(self.model, self.data)
            glfw.set_key_callback(self.window, self._key)
            glfw.set_cursor_pos_callback(self.window, self._cursor)
            glfw.set_scroll_callback(self.window, self._scroll)
            return self
        except BaseException:
            self.close()
            raise

    def _key(self, window, key, scancode, action, modifiers):
        if action == self.glfw.PRESS:
            self.key_callback(key)

    def _cursor(self, window, x, y):
        previous, self.last_cursor = self.last_cursor, (x, y)
        if previous is None:
            return
        glfw, mj = self.glfw, self.mujoco
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if not (left or right or middle):
            return
        height = max(1, glfw.get_window_size(window)[1])
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, (x-previous[0])/height,
                         (y-previous[1])/height, self.scene, self.cam)

    def _scroll(self, window, xoffset, yoffset):
        self.mujoco.mjv_moveCamera(self.model, self.mujoco.mjtMouse.mjMOUSE_ZOOM,
                                  0.0, -0.05*yoffset, self.scene, self.cam)

    def lock(self):
        # Configuration and rendering are on the same main thread.
        return nullcontext()

    def is_running(self):
        return not self.closed and bool(self.window) and not self.glfw.window_should_close(self.window)

    def sync(self):
        glfw, mj = self.glfw, self.mujoco
        if self.is_running():
            width, height = glfw.get_framebuffer_size(self.window)
            if width > 0 and height > 0:
                mj.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam,
                                   mj.mjtCatBit.mjCAT_ALL, self.scene)
                mj.mjr_render(mj.MjrRect(0, 0, width, height), self.scene, self.context)
                glfw.swap_buffers(self.window)
            glfw.poll_events()

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Free MuJoCo's GPU resources while this owned context is still current.
        # Always attempt window/library teardown, including on rendering errors.
        try:
            if self.context is not None:
                try:
                    self.context.free()
                finally:
                    self.context = None
        finally:
            self.scene = None
            try:
                if self.window:
                    self.glfw.destroy_window(self.window)
            finally:
                self.window = None
                if self.initialized:
                    self.initialized = False
                    self.glfw.terminate()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def launch(model, data, *, key_callback):
    return TeachingViewer(model, data, key_callback=key_callback)
