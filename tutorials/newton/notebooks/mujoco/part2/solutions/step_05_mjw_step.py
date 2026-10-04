# Step 5 — advance the physics on the GPU.
#
# This single call replaces mujoco.mj_step and is the entire point of Part 2.
# It advances EVERY world in the batch by one timestep, so its cost is close to
# flat as nworld grows until the GPU saturates.
#
# Under the hood mjw.step is dozens of Warp kernel launches. Calling it
# directly, as here, pays that launch overhead every time — which is why
# Step 7 captures it into a CUDA graph for the throughput path.

mjw.step(m, d)
