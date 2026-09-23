# Step 3 — seed the device state from the host, then run a forward pass.
#
# Two conversions happen on every one of these lines:
#
#   shape  (nq,)  ->  (1, nq)     mjd.qpos[None, :] adds the leading world axis
#   dtype  float64 -> float32     MJWarp is single precision; MuJoCo defaults
#                                 to double, so expect small numerical drift
#
# mjw.forward() is the MJWarp equivalent of mujoco.mj_forward: it computes all
# quantities derived from qpos/qvel (poses, Jacobians, contacts) without
# advancing time, so the first step starts from a consistent state.

wp.copy(d.qpos, wp.array(mjd.qpos[None, :], dtype=wp.float32, device=device))
wp.copy(d.qvel, wp.array(mjd.qvel[None, :], dtype=wp.float32, device=device))
wp.copy(d.ctrl, wp.array(mjd.ctrl[None, :], dtype=wp.float32, device=device))
mjw.forward(m, d)
