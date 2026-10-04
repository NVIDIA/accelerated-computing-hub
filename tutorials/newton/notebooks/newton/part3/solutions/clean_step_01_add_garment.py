# SPDX-License-Identifier: MIT
# Step 1: paste this function into clean_table_scene.py.
# Imports and scene helpers are supplied by the exercise file.

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
