# Step 1 — compile the model and allocate its mutable state.
#
# MjModel stores model parameters (some are mutable) and may be shared: geometry, inertia, joint structure, actuator
# limits. MjData is the per-simulation state: qpos, qvel, ctrl, contacts, time.
#
# We call load_pick_place_model() instead of mujoco.MjModel.from_xml_path() so
# that every part of the series simulates the identical model (it raises the
# arm actuator force limit — see pick_place_common.py for why).

model = load_pick_place_model(xml_path)
data = mujoco.MjData(model)
