# Step 6 — mirror world 0 back to the host.
#
# .numpy() copies a Warp array from device memory into a NumPy array, and the
# [0] index selects world 0 out of the (nworld, nq) batch.
#
# Migration-check mode uses CPU state for the MuJoCo passive viewer and final
# task inspection. Waypoint IK runs separately on a command-based scratch state;
# this controller does not feed measured GPU positions back into the IK solve.
#
# This is a synchronisation point: .numpy() blocks until the GPU finishes. In a
# real training loop you would never do this per substep — you would keep
# observations on device and hand them straight to the policy.

mjd.qpos[:] = d.qpos.numpy()[0]
mjd.qvel[:] = d.qvel.numpy()[0]
