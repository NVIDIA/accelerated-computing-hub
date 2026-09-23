# Step 2 — allocate batched device state.
#
# make_data() is where the GPU asks for something MuJoCo never did: explicit
# memory budgets, fixed before the first step.
#
#   nworld  - number of parallel worlds. Every Data field gains this as a
#             leading dimension, so d.qpos has shape (nworld, nq).
#   nconmax - expected contacts PER WORLD. The global budget is
#             nworld * nconmax, and one world may exceed its share as long as
#             the total does not. (naconmax sets the global cap directly.)
#   njmax   - maximum constraints per world. Unlike nconmax this is a HARD
#             per-world limit.
#
# Note these are not read from the deprecated MJCF size/nconmax and size/njmax
# fields: you must pass them here.
#
# Undersize them and you get runtime overflow warnings; oversize them and you
# waste VRAM that could have held more worlds. Tune with
# `mjwarp-testspeed <scene.xml> --measure_alloc`.

d = mjw.make_data(mjm, nworld=1, nconmax=nconmax, njmax=njmax)
