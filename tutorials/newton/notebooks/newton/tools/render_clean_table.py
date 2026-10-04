#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Render verified Newton recordings as figures and animated replays.

The plot renderer shows collision geometry; the GL renderer uses the matching
robot visual meshes. Neither advances the simulation. A successful matching
JSON task report is required before export.
The report must bind the exact NPZ bytes with record_sha256; old recordings
without that binding must be rerun, never retroactively certified.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from itertools import product
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part3"))
from final_check import validate_clean_table_report


BOX_CORNERS = np.asarray(list(product((-1.0, 1.0), repeat=3)))
BOX_FACES = np.asarray([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])


def story_frames(data):
    """Select measured carries, never infer a grasp from a fixed timestamp."""
    times = data["time"]
    phases, objects = data["phase"], data["active_object"]
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Recording times must be strictly increasing")
    if phases.shape != times.shape or objects.shape != times.shape:
        raise ValueError("Recorded phase/object labels must align with every state")
    frames = [(0, "Objects on the table")]
    carries = []
    for name, label in (("red_cube", "Red cube"), ("blue_cube", "Blue cube"),
                        ("shirt", "Cloth"), ("cable", "Cable")):
        indices = np.flatnonzero((phases == "carry") & (objects == name))
        if not len(indices):
            raise ValueError(f"Recording contains no carry frames for {name}")
        carries.append((int(indices[len(indices) // 2]), f"{label}: held during transport"))
    frames.extend(sorted(carries, key=lambda item: item[0]))
    frames.append((len(times) - 1, "All four objects released and settled"))
    return frames


def scene_limits(data):
    """Keep one camera volume around the recorded robot and material motion."""
    points = np.concatenate((data["body_q"][..., :3].reshape(-1, 3),
                             data["particle_q"].reshape(-1, 3),
                             np.asarray([data["bin_lower"], data["bin_upper"]])))
    lower, upper = points.min(axis=0) - 0.07, points.max(axis=0) + 0.07
    lower[2] = min(0.0, lower[2])
    return lower, upper


def render_gl_story(data, report, output_dir, frames, *, gif=False):
    """Replay verified poses with the matching official robot's visual meshes.

    Rebuilding provides geometry only. No solver or controller advances the
    scene, and every displayed dynamic pose comes from the bound recording.
    """
    import warp as wp
    import newton
    from newton.viewer import ViewerGL
    from PIL import Image, ImageDraw
    from newton_assets import NEWTON_ASSETS_REF

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part3" / "solutions"))
    from clean_table_scene_solution import build_scene

    if report.get("newton") != newton.__version__ or report.get("newton_assets_ref") != NEWTON_ASSETS_REF:
        raise SystemExit("GL replay needs the recording's Newton version and pinned robot assets")
    wp.init()
    # CPU readback works with an ordinary headless OpenGL context; rendering
    # does not require CUDA/OpenGL interoperability or another physics run.
    with wp.ScopedDevice("cpu"):
        scene = build_scene(report["robot"], device="cpu")
        model, state = scene.model, scene.model.state()
        for name in ("shape_body", "shape_type", "shape_scale", "shape_transform",
                     "tri_indices", "joint_parent", "joint_child"):
            current, recorded = getattr(model, name).numpy(), data[name]
            matches = current.shape == recorded.shape and np.allclose(current, recorded, rtol=0, atol=1e-7)
            if name == "shape_transform" and current.shape == recorded.shape:
                # USD can choose q or -q on different platforms. They describe
                # the same rotation; positions still have to agree.
                rotation_error = np.minimum(np.linalg.norm(current[:, 3:] - recorded[:, 3:], axis=1),
                                            np.linalg.norm(current[:, 3:] + recorded[:, 3:], axis=1))
                matches = (np.allclose(current[:, :3], recorded[:, :3], rtol=0, atol=1e-7)
                           and np.all(rotation_error <= 1e-6))
            if not matches:
                raise SystemExit(f"GL replay model differs from the recording: {name}")
        if data["body_q"].shape[1:] != (model.body_count, 7) or data["particle_q"].shape[1:] != (model.particle_count, 3):
            raise SystemExit("GL replay state dimensions do not match the reconstructed model")

        # Imported reference sites are debugging markers, not physical parts
        # of the gripper. Hide those markers in this display-only model.
        flags = model.shape_flags.numpy()
        flags[(flags & int(newton.ShapeFlags.SITE)) != 0] &= ~int(newton.ShapeFlags.VISIBLE)
        model.shape_flags.assign(flags)

        viewer = ViewerGL(width=960, height=640, headless=True,
                          enable_cuda_interop=ViewerGL.CudaInterop.NONE)
        try:
            viewer.show_static = False
            viewer.set_model(model)
            lower, upper = scene_limits(data)
            target = (lower + upper) / 2
            distance = max(0.75, float(np.max(upper - lower)) * 1.65)
            pitch, yaw = np.radians(-28.0), np.radians(125.0)
            direction = np.asarray([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
            def capture(frame, *, close=False, overview=False):
                state.body_q.assign(data["body_q"][frame])
                state.particle_q.assign(data["particle_q"][frame])
                camera_target, camera_distance = target, distance
                if close:
                    # Show the physical grip clearly in the four transport
                    # panels. Overview panels and the animation keep one view.
                    payload = scene.payload_points(state)[str(data["active_object"][frame])]
                    camera_target = (payload.min(axis=0) + payload.max(axis=0)) / 2
                    camera_distance = max(0.35, float(np.max(np.ptp(payload, axis=0))) * 2.5)
                elif overview:
                    points = np.vstack((data["body_q"][frame, :, :3], data["particle_q"][frame],
                                        data["bin_lower"], data["bin_upper"]))
                    camera_target = (points.min(axis=0) + points.max(axis=0)) / 2
                    camera_distance = max(0.80, float(np.max(np.ptp(points, axis=0))) * 1.7)
                viewer.set_camera(pos=wp.vec3(*(camera_target - camera_distance * direction)),
                                  pitch=-28.0, yaw=125.0)
                viewer.begin_frame(float(data["time"][frame]))
                viewer.log_state(state)
                viewer.end_frame()
                return viewer.get_frame().numpy()

            # A portrait montage retains readable labels when embedded at a
            # document's 6–6.5 inch text width (14 pt becomes about 9 pt).
            fig, axes = plt.subplots(3, 2, figsize=(10, 11.5), facecolor="white")
            for panel, ((frame, title), ax) in enumerate(zip(frames, axes.flat)):
                pixels = capture(frame, close=0 < panel < len(frames) - 1,
                                 overview=panel in (0, len(frames) - 1))
                Image.fromarray(pixels).save(output_dir / f"clean-table-{report['robot']}-frame-{panel + 1}.png")
                ax.imshow(pixels)
                ax.set_title(f"{title}\nt = {data['time'][frame]:.1f} s", loc="left", fontsize=14)
                ax.axis("off")
            fig.suptitle(f"Newton pick-and-place · {report['robot']}\nRecorded states with official robot geometry", fontsize=18)
            # Reserve real space for the two-line titles. Tight layout can
            # overlap image rows when their axes have been hidden.
            fig.subplots_adjust(left=0.025, right=0.975, bottom=0.015, top=0.88,
                                wspace=0.10, hspace=0.32)
            figure = output_dir / f"clean-table-{report['robot']}.png"
            fig.savefig(figure, dpi=150, facecolor="white")
            plt.close(fig)
            print(figure.resolve())
            if gif:
                stride = max(1, int(np.ceil(len(data["time"]) / 240)))
                selected = list(range(0, len(data["time"]), stride))
                if selected[-1] != len(data["time"]) - 1:
                    selected.append(len(data["time"]) - 1)
                images = []
                for frame in selected:
                    picture = Image.fromarray(capture(frame))
                    drawing = ImageDraw.Draw(picture)
                    phase = str(data["phase"][frame]).replace("_", " ")
                    payload = str(data["active_object"][frame]).replace("_", " ")
                    drawing.rectangle((0, 0, picture.width, 34), fill=(27, 30, 38))
                    drawing.text((12, 10), f"{report['robot']} | {payload} {phase} | t={data['time'][frame]:.1f}s | accelerated replay",
                                 fill="white")
                    images.append(picture.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
                animation = output_dir / f"clean-table-{report['robot']}.gif"
                images[0].save(animation, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=False)
                print(animation.resolve())
        finally:
            viewer.close()


def transform(pose, points):
    x, y, z, w = pose[3:7]
    rotation = np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return np.asarray(points) @ rotation.T + pose[:3]


def rounded_faces(radius, half_height=0.0, cylinder=False):
    angles = np.linspace(0.0, 2*np.pi, 13)
    if cylinder:
        rings = [(radius, -half_height), (radius, half_height)]
    else:
        polar = np.linspace(0.0, np.pi, 9)
        rings = [(radius*np.sin(t), radius*np.cos(t)+(half_height if t <= np.pi/2 else -half_height)) for t in polar]
        # Include the lower cylinder rim even for an even-length polar list.
        rings.insert(5, (radius, -half_height))
    vertices = np.asarray([[(r*np.cos(a), r*np.sin(a), z) for a in angles] for r, z in rings])
    return np.asarray([[vertices[j,i],vertices[j,i+1],vertices[j+1,i+1],vertices[j+1,i]]
                       for j in range(len(rings)-1) for i in range(len(angles)-1)])


def draw(ax, data, frame, title, limits=None):
    q = data["body_q"][frame]
    # Draw the actual analytic collision shapes. Mesh-only links are indicated
    # by the articulation skeleton rather than invented visual geometry.
    for shape, kind in enumerate(data["shape_type"]):
        scale = data["shape_scale"][shape]
        if kind == 7:  # Newton GeoType.BOX
            faces = (BOX_CORNERS*scale)[BOX_FACES]
        elif kind == 3:
            faces = rounded_faces(scale[0])
        elif kind in (4, 6):
            faces = rounded_faces(scale[0], scale[1], cylinder=kind == 6)
        else:
            continue
        faces = transform(data["shape_transform"][shape], faces.reshape(-1,3)).reshape(faces.shape)
        body = int(data["shape_body"][shape])
        if body >= 0:
            faces = transform(q[body], faces.reshape(-1,3)).reshape(faces.shape)
        color = np.clip(data["shape_color"][shape], 0, 1)
        artist = Poly3DCollection(faces, facecolor=color, edgecolor=(0.1,0.15,0.2,0.22), linewidth=0.12,
                                  alpha=0.24 if body < 0 else 0.95)
        ax.add_collection3d(artist)
    for parent, child in zip(data["joint_parent"], data["joint_child"]):
        if parent >= 0 and child >= 0:
            points = q[[int(parent),int(child)],:3]
            ax.plot(*points.T, color="#667581", linewidth=2.2)
    cloth = data["particle_q"][frame]
    ax.add_collection3d(Poly3DCollection(cloth[data["tri_indices"]], facecolor="#42bcb0",
                                        edgecolor="#246c71", linewidth=0.25, alpha=1.0))
    lower, upper = scene_limits(data) if limits is None else limits
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
           xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_proj_type("ortho")
    ax.set_box_aspect(upper - lower)
    ax.view_init(elev=27, azim=-117)
    ax.set_title(f"{title}\nt = {data['time'][frame]:.1f} s", fontsize=11, loc="left", pad=8)
    ax.tick_params(labelsize=7, pad=0)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill = False
        axis.set_major_locator(MaxNLocator(nbins=3))
    ax.grid(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--renderer", choices=("plot", "gl"), default="plot",
                        help="Analytic collision plots, or Newton GL replay with the matching robot meshes")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("success") is not True:
        raise SystemExit("Refusing to publish a rollout that did not pass its task check")
    valid, detail = validate_clean_table_report(report, robot=report.get("robot"))
    if not valid:
        raise SystemExit(f"Refusing to publish an unverified gripper task: {detail}")
    expected_hash = report.get("record_sha256")
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise SystemExit("Missing or invalid record_sha256 (SHA-256); rerun with --record and --report")
    record_bytes = args.record.read_bytes()
    if hashlib.sha256(record_bytes).hexdigest() != expected_hash:
        raise SystemExit("SHA-256 mismatch: the recording does not belong to this result report")
    # Decode exactly the bytes checked above, not a second potentially replaced file.
    data = np.load(io.BytesIO(record_bytes), allow_pickle=False)
    if not np.isclose(data["time"][-1], report["simulation_seconds"]):
        raise SystemExit("The trajectory and result report do not describe the same duration")
    if not all(np.isfinite(data[name]).all() for name in ("time", "body_q", "particle_q")):
        raise SystemExit("The recorded state is nonfinite")
    try:
        frames = story_frames(data)
    except (KeyError, ValueError) as error:
        raise SystemExit(f"Incomplete pick-and-place recording: {error}") from error
    limits = scene_limits(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot = report["robot"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.labelsize":8})
    if args.renderer == "gl":
        render_gl_story(data, report, args.output_dir, frames, gif=args.gif)
        return
    fig = plt.figure(figsize=(14,9), facecolor="white")
    for i, (frame, title) in enumerate(frames):
        draw(fig.add_subplot(2,3,i+1, projection="3d"), data, frame, title, limits)
    fig.suptitle(f"Newton pick-and-place · {robot}\nMuJoCo robot and cubes ↔ VBD cloth and cable", fontsize=15, y=0.98)
    fig.legend(handles=[Patch(color="#42bcb0",label="Cloth mesh"), Patch(color="#f29419",label="Cable (capsule rod)"),
                        Patch(color="#667581",label="Robot articulation")], loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(top=.85,bottom=.07,hspace=.2,wspace=.04,left=.02,right=.98)
    figure = args.output_dir/f"clean-table-{robot}.png"
    fig.savefig(figure,dpi=150,facecolor="white")
    plt.close(fig)
    print(figure.resolve())
    if args.gif:
        fig = plt.figure(figsize=(8,5),facecolor="white")
        ax = fig.add_subplot(111,projection="3d")
        stride = max(1, int(np.ceil(len(data["time"]) / 240)))
        animation_frames = list(range(0, len(data["time"]), stride))
        if animation_frames[-1] != len(data["time"]) - 1:
            animation_frames.append(len(data["time"]) - 1)
        interval = float(np.median(np.diff(data["time"]))) * stride
        playback_speed = interval * 10
        def update(frame):
            ax.clear()
            phase = str(data["phase"][frame]).replace("_", " ")
            payload = str(data["active_object"][frame]).replace("_", " ")
            label = f"Newton · {robot} · {payload} {phase}".strip()
            draw(ax,data,frame,f"{label}\nRecorded collision geometry · {playback_speed:.1f}× playback", limits)
            return []
        animation = FuncAnimation(fig,update,frames=animation_frames,interval=100,blit=False)
        gif = args.output_dir/f"clean-table-{robot}.gif"
        animation.save(gif,writer=PillowWriter(fps=10),dpi=90)
        plt.close(fig)
        print(gif.resolve())


if __name__ == "__main__":
    main()
