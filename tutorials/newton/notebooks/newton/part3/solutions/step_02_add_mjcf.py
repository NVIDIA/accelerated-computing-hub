# Step 2 — import the robot.
#
# add_mjcf() parses the same MJCF file Parts 1 and 2 used. The equivalent calls
# add_urdf() and add_usd() exist too, which is one of the main reasons to route
# through Newton rather than calling MJWarp directly.
#
#   ignore_names=["floor"]        skip the MJCF ground; Newton adds its own
#                                 plane in Step 5, and two overlapping ground
#                                 planes produce duplicate contacts
#   collapse_fixed_joints=True    merge bodies joined by fixed welds into one
#                                 rigid body. Fewer bodies means a smaller
#                                 articulation and a faster solve, with no
#                                 change in physical behaviour.

builder.add_mjcf(
    str(robot_xml),
    ignore_names=["floor"],
    collapse_fixed_joints=True,
)
