# Step 5 — ground plane, then finalize.
#
# finalize() is the boundary between authoring and simulating. It packs every
# body, shape, joint and material into flat Warp arrays on the device and
# returns a newton.Model. After this call the topology is fixed:
# you can change values (masses, gains, targets) but not add bodies.
#
# This mirrors the MuJoCo split you already know — MjSpec/MJCF for authoring,
# MjModel for simulating — with the difference that the source format can be
# MJCF, URDF or OpenUSD.

builder.add_ground_plane()

# Inside build_newton_model() this is simply:
#     return builder.finalize()
model = builder.finalize()
