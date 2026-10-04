# SPDX-License-Identifier: MIT
# Step 2: paste this function into clean_table_scene.py.
# Imports and scene helpers are supplied by the exercise file.

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
