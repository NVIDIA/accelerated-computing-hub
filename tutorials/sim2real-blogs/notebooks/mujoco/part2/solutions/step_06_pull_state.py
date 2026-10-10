# Step 6 — mirror world 0 back to the host.
#
# .numpy() copies a Warp array from device memory into a NumPy array, and the
# [0] index selects world 0 out of the (nworld, nq) batch.
#
# We need this in parity mode because two consumers still live on the CPU: the
# damped-least-squares IK solve, and the MuJoCo passive viewer. Both read from
# mjd, so mjd has to track what the GPU computed.
#
# This is a synchronisation point: .numpy() blocks until the GPU finishes. In a
# real training loop you would never do this per substep — you would keep
# observations on device and hand them straight to the policy.

mjd.qpos[:] = d.qpos.numpy()[0]
mjd.qvel[:] = d.qvel.numpy()[0]
