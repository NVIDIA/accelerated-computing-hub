# SPDX-License-Identifier: MIT
# Step 4: paste this method into Example.

def simulate(self):
    for _ in range(self.sim_substeps):
        self.state_0.clear_forces()
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._sample_contacts()
