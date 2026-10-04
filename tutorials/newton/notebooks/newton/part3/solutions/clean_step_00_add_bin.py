# SPDX-License-Identifier: MIT
# Step 0: paste this function into clean_table_scene.py.
# Imports and scene helpers are supplied by the exercise file.

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
