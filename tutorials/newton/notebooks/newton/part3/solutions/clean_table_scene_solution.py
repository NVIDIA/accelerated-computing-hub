# SPDX-FileCopyrightText: Copyright (c) 2026 Johnny Nuñez Cano
# SPDX-License-Identifier: MIT
"""Gripper pick-and-place scene: MuJoCo robot/cubes, VBD free cloth/rod."""
from __future__ import annotations
import sys
from pathlib import Path
import mujoco
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pick_place_common as task
from robots import get_robot
from newton_assets import add_robot_usd
from clean_table_task import Bin, CleanScene, GripperIK, CheckedCollisionPipeline, RIGID_CONTACTS_PER_BODY, SOFT_CONTACTS_PER_BODY


def add_bin(builder, x, table_z):
    """An open receiving box with four physical walls and a floor."""
    lower = np.asarray([x - 0.115, 0.055, table_z + 0.006])
    upper = np.asarray([x + 0.115, 0.245, table_z + 0.091])
    thickness = 0.006
    center, half = (lower + upper) / 2, (upper - lower) / 2
    cfg = newton.ModelBuilder.ShapeConfig(density=0., ke=3000., kd=1., mu=0.8, gap=0.)
    pieces = [
        ('floor', (center[0], center[1], lower[2] - thickness / 2), (half[0], half[1], thickness / 2)),
        ('left', (lower[0] - thickness / 2, center[1], center[2]), (thickness / 2, half[1] + thickness, half[2])),
        ('right', (upper[0] + thickness / 2, center[1], center[2]), (thickness / 2, half[1] + thickness, half[2])),
        ('near', (center[0], lower[1] - thickness / 2, center[2]), (half[0], thickness / 2, half[2])),
        ('far', (center[0], upper[1] + thickness / 2, center[2]), (half[0], thickness / 2, half[2])),
    ]
    shapes = []
    for label, pos, size in pieces:
        shapes.append(builder.add_shape_box(body=-1, xform=wp.transform(pos, wp.quat_identity()),
                     hx=size[0], hy=size[1], hz=size[2], cfg=cfg, color=(0.16, 0.35, 0.48), label='bin_' + label))
    return Bin(lower, upper), shapes


def add_garment(builder, x, table_z):
    """Free shirt silhouette with a raised, precreased central fold to grasp.

    Every vertex has mass and every fold is elastic. No particle is pinned to a
    jaw or to the table. This is a small cloth patch, not a calibrated garment.
    """
    cells = [(i, j) for i in range(-5, 5) for j in range(-5, 5)
             if ((-3 <= i < 3 and -5 <= j < 3) or 1 <= j < 4) and not (-1 <= i < 1 and j >= 3)]
    nodes = sorted({p for i, j in cells for p in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))})
    ids = {p: i for i, p in enumerate(nodes)}
    # An initially folded sheet with a 20 mm ridge remains graspable above the
    # tabletop while retaining real bending and stretching degrees of freedom.
    vertices = [(i * 0.009, j * 0.006, 0.003 + 0.021 * max(0., 1. - abs(j) / 3.)) for i, j in nodes]
    triangles = []
    for i, j in cells:
        a, b, c, d = [ids[p] for p in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))]
        triangles.extend((a, b, c, b, d, c))
    builder.add_cloth_mesh(pos=wp.vec3(x - 0.005, -0.085, table_z), rot=wp.quat_identity(),
        scale=1., vel=wp.vec3(0.), vertices=vertices, indices=triangles,
        density=0.35, tri_ke=600., tri_ka=600., tri_kd=0.001,
        edge_ke=0.02, edge_kd=0.0001, particle_radius=0.002,
        validate_mesh=True, label='shirt')


def add_cable(builder, x, table_z, *, offset=(0.075, -0.015)):
    """A short freely moving curved rod, with a 14 mm physical diameter."""
    # Segment length exceeds the capsule diameter, avoiding penetration between
    # second-neighbour endcaps (only immediately adjacent bodies are filtered).
    angle = np.linspace(0., np.pi, 7)
    positions = np.column_stack((x + offset[0] + 0.035 * np.cos(angle),
                                 np.full(len(angle), offset[1]),
                                 table_z + 0.0075 + 0.035 * np.sin(angle)))
    shape_start = builder.shape_count
    bodies, joints = builder.add_rod(rod=newton.Rod(points=positions, radius=0.007), body_frame_origin='com',
        stretch_stiffness=1.0e4, stretch_damping=0.02, bend_stiffness=0.01, bend_damping=0.0001,
        cfg=newton.ModelBuilder.ShapeConfig(density=500., ke=3000., kd=0.1, mu=1.2, gap=0.001),
        color=(0.95, 0.58, 0.10), label='cable')
    return bodies, joints, list(range(shape_start, builder.shape_count))


def build_scene(robot='so101', *, device=None):
    newton.use_coord_layout_targets = True
    spec = get_robot(robot)
    source = task.resolve_pick_place_scene(spec=spec).parent / spec.robot_xml
    ik_model = mujoco.MjModel.from_xml_path(str(source))
    ik = GripperIK(ik_model, spec)
    x = 0.305 if spec.key == 'so101' else 0.47
    table_z = spec.table_top_z
    initial_ctrl = ik.solve([x, -0.10, table_z + 0.18], spec.gripper_open, iterations=250)
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = 0.
    SolverMuJoCo.register_custom_attributes(builder)
    SolverVBD.register_custom_attributes(builder)
    add_robot_usd(builder, robot)
    robot_joints, q_indices = [], []
    for actuator in range(ik_model.nu):
        name = ik_model.joint(int(ik_model.actuator_trnid[actuator, 0])).name
        matches = [j for j, label in enumerate(builder.joint_label) if label.split('/')[-1] == name]
        if len(matches) != 1:
            raise ValueError(f'Cannot map actuator joint {name}: {matches}')
        joint = matches[0]
        robot_joints.append(joint)
        q, d = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        q_indices.append(q)
        builder.joint_q[q] = float(initial_ctrl[actuator])
        builder.joint_target_q[q] = float(initial_ctrl[actuator])
        builder.joint_target_mode[d] = int(newton.JointTargetMode.POSITION)
        builder.joint_target_ke[d], builder.joint_target_kd[d] = spec.arm_target_ke, spec.arm_target_kd
        if spec.key == 'rebot' and actuator in spec.gripper_ctrl_indices:
            # The imported linear fingers need their authored drive stiffness;
            # applying rotary-arm gains here makes the native cube pinch soft.
            builder.joint_target_ke[d] = float(ik_model.actuator_gainprm[actuator, 0])
            builder.joint_target_kd[d] = float(-ik_model.actuator_biasprm[actuator, 2])
        if actuator not in spec.gripper_ctrl_indices:
            builder.joint_effort_limit[d] = spec.arm_effort_limit
            if spec.arm_force_limit is not None:
                builder.custom_attributes['mujoco:actuator_forcerange'].values[actuator] = wp.vec2(-spec.arm_force_limit, spec.arm_force_limit)

    def surviving_body(mj_body):
        while mj_body:
            name = ik_model.body(mj_body).name
            matches = [i for i, label in enumerate(builder.body_label) if label.split('/')[-1] == name]
            if len(matches) == 1:
                return matches[0], mj_body
            mj_body = int(ik_model.body_parentid[mj_body])
        raise ValueError('A gripper body has no surviving Newton parent')

    if spec.left_jaw_geoms:
        mj_jaws = [int(ik_model.geom_bodyid[ik_model.geom(group[0]).id])
                   for group in (spec.left_jaw_geoms, spec.right_jaw_geoms)]
    else:
        mj_jaws = [ik_model.body(spec.left_jaw_body).id, ik_model.body(spec.right_jaw_body).id]
    jaw_bodies = tuple(surviving_body(b)[0] for b in mj_jaws)
    jaw_shapes = tuple([s for s, b in enumerate(builder.shape_body)
                       if b == jaw and builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES)]
                      for jaw in jaw_bodies)
    for s in range(builder.shape_count):
        builder.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if s in jaw_shapes[0] or s in jaw_shapes[1]:
            builder.shape_flags[s] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
            # MuJoCo already uses convex support geometry for imported meshes.
            if builder.shape_type[s] == newton.GeoType.MESH:
                builder.shape_type[s] = newton.GeoType.CONVEX_MESH
            builder.shape_material_mu[s] = 1.2
    proxy_joints = [robot_joints[a] for a in spec.gripper_ctrl_indices]
    proxy_bodies = set(jaw_bodies)
    for j in proxy_joints:
        proxy_bodies.update(b for b in (builder.joint_parent[j], builder.joint_child[j]) if b >= 0)
    grasp_sites = {}
    for kind in ('cube', 'material'):
        ik.material = kind == 'material'
        world_points = ik.jaw_points()
        sites = []
        for mj_body, point in zip(mj_jaws, world_points):
            newton_body, ancestor = surviving_body(mj_body)
            local = ik.data.xmat[ancestor].reshape(3, 3).T @ (point - ik.data.xpos[ancestor])
            sites.append((newton_body, local))
        grasp_sites[kind] = tuple(sites)
    table_shape = builder.add_shape_box(body=-1, xform=wp.transform((x, 0., table_z - 0.012), wp.quat_identity()),
        hx=0.18, hy=0.28, hz=0.012,
        cfg=newton.ModelBuilder.ShapeConfig(density=0., ke=3000., kd=1., mu=0.8, gap=0.),
        color=(0.44, 0.40, 0.33), label='table')
    bin_box, bin_shapes = add_bin(builder, x, table_z)
    cube_shapes = {}
    for name, dx, color in (('red_cube', -0.035, (0.85, 0.1, 0.08)), ('blue_cube', 0.045, (0.08, 0.25, 0.9))):
        body = builder.add_body(xform=wp.transform((x + dx, -0.15, table_z + spec.cube_half), wp.quat_identity()),
            mass=0.08, inertia=wp.mat33(np.eye(3) * 0.08 * (2 * spec.cube_half)**2 / 6), lock_inertia=True, label=name)
        cube_shapes[name] = builder.add_shape_box(body=body, hx=spec.cube_half, hy=spec.cube_half, hz=spec.cube_half,
            cfg=newton.ModelBuilder.ShapeConfig(density=0., ke=3000., kd=0.1, mu=1.2, gap=0.), color=color, label=name)
        proxy_bodies.add(body)
    # VBD's force-space damping is not a MuJoCo time constant. Preserve the
    # deformable material coefficients while authoring critically damped native
    # rigid contacts explicitly, in MuJoCo's (time constant, damping ratio) form.
    for shape in [table_shape, *bin_shapes, *cube_shapes.values()]:
        builder.custom_attributes['mujoco:solref'].values[shape] = wp.vec2(0.01, 1.0)
        builder.custom_attributes['mujoco:solref_mode'].values[shape] = 1
    rigid_bodies, rigid_joints = list(range(builder.body_count)), list(range(builder.joint_count))
    # The larger reBot hand needs a pickup lane clear of the bin's near wall.
    # This changes only the authored free-rod layout, never a simulated pose.
    cable_offset = (0.060, -0.205) if spec.key == 'rebot' else (0.075, -0.015)
    cable_bodies, cable_joints, cable_shapes = add_cable(builder, x, table_z, offset=cable_offset)
    add_garment(builder, x, table_z)
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 3000., 0.1, 1.2
    slots = {'red_cube': np.asarray([x - 0.065, 0.11]), 'blue_cube': np.asarray([x + 0.005, 0.11]),
             'shirt': np.asarray([x - 0.025, 0.17]), 'cable': np.asarray([x + 0.07, 0.17])}
    return CleanScene(model=model, spec=spec, bin=bin_box, bin_shapes=bin_shapes, cube_shapes=cube_shapes,
        cable_bodies=cable_bodies, cable_joints=cable_joints, cable_shapes=cable_shapes,
        rigid_bodies=rigid_bodies, rigid_joints=rigid_joints, clutter_bodies=cable_bodies, clutter_joints=cable_joints,
        target_indices=model.joint_target_q_start.numpy()[robot_joints], robot_q_indices=np.asarray(q_indices),
        initial_ctrl=initial_ctrl, ik_model=ik_model, ik_data=ik.data, table_z=table_z,
        jaw_bodies=jaw_bodies, jaw_shapes=jaw_shapes, proxy_bodies=sorted(proxy_bodies), proxy_joints=proxy_joints,
        grasp_sites=grasp_sites, slots=slots, initial_grasp_points={})


def make_clutter_pipeline(model):
    """VBD sees cable versus static/robot/cubes; cloth has its soft contacts.

    Source-owned rigid contacts remain solely in MuJoCo. This prevents a second
    solver from resolving the native cube grasp or robot/table contact twice.
    """
    cable_shapes = {s for s, label in enumerate(model.shape_label) if label.startswith('cable')}
    pairs = [(int(a), int(b)) for a, b in model.shape_contact_pairs.numpy()
             if int(a) in cable_shapes or int(b) in cable_shapes]
    return CheckedCollisionPipeline(model, broad_phase='explicit', soft_contact_gap=0.002,
        shape_pairs_filtered=wp.array(np.asarray(pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i, device=model.device))


class ObservedVBD(SolverVBD):
    """Retain the contact-native proxy reaction for read-only grasp telemetry."""
    def coupling_harvest_proxy_wrenches(self, body_local_to_proxy_global, out_body_f, **kwargs):
        super().coupling_harvest_proxy_wrenches(body_local_to_proxy_global, out_body_f, **kwargs)
        if not hasattr(self, 'last_raw_body_f'):
            self.last_raw_body_f = wp.empty_like(out_body_f)
        wp.copy(self.last_raw_body_f, out_body_f)


def build_coupled_solver(scene, *, vbd_iterations=20, coupling_iterations=2):
    model = scene.model
    return SolverCoupledProxy(model=model, entries=[
        SolverCoupled.Entry(name='rigid', bodies=scene.rigid_bodies, joints=scene.rigid_joints,
            solver=lambda view: SolverMuJoCo(view, use_mujoco_cpu=not model.device.is_cuda,
                use_mujoco_contacts=True, solver='newton', integrator='implicitfast', cone='elliptic',
                njmax=2048, nconmax=1024, iterations=100, ls_iterations=50, impratio=100)),
        SolverCoupled.Entry(name='deformable', bodies=scene.cable_bodies, joints=scene.cable_joints,
            particles=list(range(model.particle_count)),
            solver=lambda view: ObservedVBD(view, iterations=vbd_iterations, rigid_compliant_alm=True,
                particle_enable_self_contact=True, particle_self_contact_margin=0.002,
                particle_self_contact_gap=0.002, rigid_body_contact_buffer_size=RIGID_CONTACTS_PER_BODY,
                rigid_body_particle_contact_buffer_size=SOFT_CONTACTS_PER_BODY, rigid_contact_history=False)),
    ], coupling=SolverCoupledProxy.Config(iterations=coupling_iterations, proxies=[
        SolverCoupledProxy.Proxy(source='rigid', destination='deformable', bodies=scene.proxy_bodies,
            joints=scene.proxy_joints, mass_scale=1., mode='lagged', proxy_relaxation=0.5,
            collision_pipeline=make_clutter_pipeline, collide_interval=1)
    ]))
