# Step 4 — add the table and the two cubes.
#
# The distinction that matters: a shape attached to body=-1 belongs to the
# world and is static. A shape attached to a body created with add_body() is
# dynamic and gets a full 6-DoF free motion.
#
# ShapeConfig carries the contact material:
#   ke        contact stiffness      kd   contact damping
#   mu        friction coefficient   density  contribution to the body's mass
#
# The MJCF specifies 0.08 kg, not density=400 (which made these cubes lighter).
# Author BODY mass and solid-box inertia explicitly. Shape density=0 and
# lock_inertia=True prevent counting shape mass again; these bodies are dynamic.

table_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0e2, mu=1.0, density=0.0, gap=0.0)
cube_cfg = newton.ModelBuilder.ShapeConfig(ke=8.0e4, kd=8.0e2, mu=1.2, density=0.0, gap=0.0)
cube_mass = 0.08
cube_inertia = wp.mat33(np.eye(3) * cube_mass * (2.0 * task.CUBE_HALF) ** 2 / 6.0)

builder.add_shape_box(
    body=-1,
    hx=float(task.TABLE_HALF_SIZE[0]),
    hy=float(task.TABLE_HALF_SIZE[1]),
    hz=float(task.TABLE_HALF_SIZE[2]),
    xform=wp.transform(wp.vec3(*task.TABLE_CENTER), wp.quat_identity()),
    cfg=table_cfg,
    color=(0.32, 0.32, 0.32),
    label="table",
)

for label, pos, color in (
    ("red_cube", task.RED_CUBE_POS, (0.85, 0.05, 0.04)),
    ("blue_cube", task.BLUE_CUBE_POS, (0.05, 0.20, 0.90)),
):
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), label=label,
        mass=cube_mass, inertia=cube_inertia, lock_inertia=True,
    )
    builder.add_shape_box(
        body=body,
        hx=task.CUBE_HALF,
        hy=task.CUBE_HALF,
        hz=task.CUBE_HALF,
        cfg=cube_cfg,
        color=color,
        label=label,
    )
