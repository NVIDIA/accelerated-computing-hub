# Step 3 — the scripted controller.
#
# PickPlaceController owns the waypoint state machine, the damped-least-squares
# IK solve, and the gradual gripper ramp. It is backend-agnostic: it only needs
# an MjModel plus a scratch MjData for forward kinematics, which is why Parts 2
# and 3 reuse it unchanged while the physics backend changes underneath.

controller = PickPlaceController()
