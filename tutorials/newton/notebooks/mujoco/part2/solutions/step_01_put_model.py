# Step 1 — upload the compiled model to the GPU.
#
# put_model() converts an mjModel into an mjw.Model whose fields are Warp
# arrays in device memory. This example keeps model parameters fixed and
# shares them across the batch, so one upload serves all worlds.
#
# If the MJCF uses a feature MJWarp does not implement yet, this call raises.
# That is deliberate and much friendlier than silently diverging physics.

m = mjw.put_model(mjm)
