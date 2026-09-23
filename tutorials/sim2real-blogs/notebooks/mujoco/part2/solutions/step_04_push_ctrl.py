# Step 4 — push the control vector from host to device.
#
# In Part 1 this was a NumPy assignment into a view MuJoCo already owned:
#     data.ctrl[: model.nu] = ctrl
#
# On the GPU there is no shared view, so the command has to be copied across
# the PCIe bus. Note the [None, :] again: d.ctrl has shape (nworld, nu).
#
# Doing this every substep is exactly the host<->device traffic that kills
# throughput. It is acceptable here only because parity mode runs one world
# and needs the CPU in the loop for IK. Throughput mode (Step 7) keeps
# everything resident on device instead.

wp.copy(d.ctrl, wp.array(mjd.ctrl[None, :], dtype=wp.float32, device=device))
