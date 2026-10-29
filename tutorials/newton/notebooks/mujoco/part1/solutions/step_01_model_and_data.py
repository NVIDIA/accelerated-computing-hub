# Step 1 — compile the model and allocate its mutable state.
#
# MjModel stores geometry, inertia, joints and actuator settings. Parameters
# such as actuator limits can be changed. MjData holds qpos, qvel, ctrl, contacts
# and time for one simulation.
#
# We call load_pick_place_model() instead of mujoco.MjModel.from_xml_path() so
# that every part of the series simulates the identical model (it raises the
# arm actuator force limit — see pick_place_common.py for why).

model = load_pick_place_model(xml_path)
data = mujoco.MjData(model)
