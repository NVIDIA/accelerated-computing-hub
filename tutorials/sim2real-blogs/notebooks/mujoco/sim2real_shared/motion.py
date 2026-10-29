# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Corporation
# SPDX-License-Identifier: MIT
"""Record simulated poses and replay them inline without advancing physics."""
from pathlib import Path

import numpy as np


class MotionRecorder:
    def __init__(self, path: Path, data, *, fps: int, xml_path: Path, robot: str, backend: str):
        self.path, self.xml_path = Path(path), Path(xml_path).resolve()
        self.robot, self.backend = robot, backend
        self.stride = max(1, round(fps / 10))
        self.poses, self.times = [data.qpos.copy()], [float(data.time)]

    def capture(self, data, frame: int) -> None:
        if frame % self.stride == 0:
            self.poses.append(data.qpos.copy())
            self.times.append(float(data.time))

    def save(self, *, success: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, qpos=np.asarray(self.poses), time=np.asarray(self.times),
                            xml_path=str(self.xml_path), robot=self.robot, backend=self.backend,
                            success=bool(success))


def playback(path: Path, *, speed: float = 2.0):
    """Return a player for a verified episode, with play/pause and frame controls.

    Frames use recorded CPU or GPU poses; mj_forward only rebuilds render state.
    No contact integration, controller update, or new simulation occurs here.
    """
    import mujoco
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    from robots import get_robot

    if speed <= 0:
        raise ValueError("Playback speed must be positive.")
    with np.load(path, allow_pickle=False) as recording:
        if not bool(recording["success"]):
            raise ValueError("This episode did not pass the physical task checks.")
        poses, times = recording["qpos"], recording["time"]
        model = mujoco.MjModel.from_xml_path(str(recording["xml_path"]))
        spec, backend = get_robot(str(recording["robot"])), str(recording["backend"])
    if len(poses) < 2 or not np.isfinite(poses).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Recording must contain finite poses at increasing simulation times.")
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = spec.camera_lookat
    camera.distance, camera.azimuth, camera.elevation = spec.camera_distance, 140, -25
    indices = list(range(0, len(poses), 2))
    if indices[-1] != len(poses) - 1:
        indices.append(len(poses) - 1)
    fps = speed / float(np.median(np.diff(times[indices])))
    figure, axis = plt.subplots(figsize=(6.4, 4.8), dpi=75)
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=.92)
    try:
        with mujoco.Renderer(model, height=360, width=480) as renderer:
            picture = axis.imshow(np.zeros((360, 480, 3), dtype=np.uint8))

            def draw(index):
                data.qpos[:] = poses[index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera)
                picture.set_data(renderer.render())
                axis.set_title(f"{spec.display_name} · {backend} · {times[index]:.1f} s · {speed:g}×")
                return (picture,)

            movie = FuncAnimation(figure, draw, frames=indices, interval=1000 / fps, repeat=False)
            with plt.rc_context({"animation.embed_limit": 64, "animation.frame_format": "jpeg"}):
                html = movie.to_jshtml(fps=fps, default_mode="once")
        return HTML(html)
    finally:
        plt.close(figure)
