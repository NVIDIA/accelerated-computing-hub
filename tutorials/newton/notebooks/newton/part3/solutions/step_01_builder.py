# Step 1 — create the builder and register the solver's custom attributes.
#
# ModelBuilder is Newton's scene description: an editable, host-side collection
# of bodies, shapes, joints and materials that you later finalize() into a
# device-resident Model.
#
# register_custom_attributes() MUST run before any asset is added. It teaches
# the builder about MJWarp-specific fields (solver reference parameters, contact
# settings, actuator metadata) so add_mjcf can carry them through instead of
# discarding them. Call it late and those attributes are silently lost.

newton.use_coord_layout_targets = True
builder = newton.ModelBuilder()
# Preserve the MJCF's zero detection gap. Newton 1.5's fallback is 0.1 m,
# which changes native MuJoCo contact generation at gripper scale.
builder.default_shape_cfg.gap = 0.0
SolverMuJoCo.register_custom_attributes(builder)
