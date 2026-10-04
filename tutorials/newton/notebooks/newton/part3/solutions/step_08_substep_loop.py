# Step 8 — the substep loop.
#
# Compare the three backends side by side:
#
#   Part 1   mujoco.mj_step(model, data)                      in-place
#   Part 2   mjw.step(m, d)                                   in-place, batched
#   Part 3   solver.step(state_in, state_out, control, ...)   out of place
#
# clear_forces() zeroes accumulated external forces so they do not carry over
# between substeps. The swap at the end is what turns an out-of-place solver
# into a loop: this substep's output becomes the next substep's input, with no
# allocation and no copying.
# Refresh Newton contact geometry at EVERY substep, not only once per frame.
# The native CPU/MuJoCo-contacts path has no external collision pipeline.
# The scaffold's diagnostic kernel retains the largest contact count on the
# device, so --test also detects truncation inside captured CUDA graphs.

for _ in range(self.sim_substeps):
    self.state_0.clear_forces()
    if self.collision_pipeline is not None:
        self.collision_pipeline.collide(self.state_0, self.contacts)
        wp.launch(_record_contact_peak, dim=1,
                  inputs=[self.contacts.rigid_contact_count, self.contact_peak],
                  device=self.model.device)
    self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
    self.state_0, self.state_1 = self.state_1, self.state_0
