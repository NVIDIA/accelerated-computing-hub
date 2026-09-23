# Step 2 — initial conditions.
#
# apply_arm_ctrl() writes both data.ctrl and the matching data.qpos, then runs
# mj_forward so all body/geom world poses become valid immediately. reset_cubes()
# snaps the two free-joint cubes to their spawn poses and zeroes their velocity.
#
# Note this "teleport" style of initialisation is only ever used to set up the
# scene. Once the loop starts, state must evolve through mj_step dynamics.

apply_arm_ctrl(model, data, spec.home_ctrl)
reset_cubes(model, data)
