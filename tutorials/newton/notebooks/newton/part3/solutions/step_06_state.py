# Step 6 — allocate Newton's state objects.
#
# MuJoCo keeps everything mutable in one MjData. Newton splits it by role:
#
#   State     positions and velocities at time t (joint_q, joint_qd, body_q...)
#   Control   actuation signals (joint_target_q, joint_f...)
#   Contacts  tensorised contact geometry produced by the collision pipeline
#
# Two states are required because solver.step() is OUT OF PLACE: it reads
# state_0 and writes state_1. That design is what makes the whole step
# suitable for GPU execution. Capture/differentiability are solver-specific,
# not guaranteed merely by allocating two states (nor tested by a CPU run).
#
# eval_fk() runs forward kinematics from the seeded joint angles so body
# transforms are valid before the first step — the Newton analogue of
# mujoco.mj_forward.

self.state_0 = self.model.state()
self.state_1 = self.model.state()
self.control = self.model.control()
newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
