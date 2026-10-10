# Step 4 — advance the physics.
#
# The control loop runs at `fps` (50 Hz) but physics needs a much smaller
# timestep for stable contact, so each control frame issues `sim_substeps`
# calls to mj_step. The controller output is held constant across substeps,
# exactly like a real servo holding a setpoint between commands.
#
# This is THE line that identifies the backend. Keep an eye on it:
#     Part 1   mujoco.mj_step(model, data)
#     Part 2   mjw.step(m, d)
#     Part 3   solver.step(state_0, state_1, control, contacts, sim_dt)

for _ in range(sim_substeps):
    data.ctrl[: model.nu] = ctrl
    mujoco.mj_step(model, data)
