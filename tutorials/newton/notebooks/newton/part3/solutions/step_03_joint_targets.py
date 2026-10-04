# Step 3 — configure position-controlled joints.
#
# This is Newton's equivalent of MuJoCo's position actuators plus the
# actuator_forcerange boost from Part 1.
#
#   joint_q            initial joint angles (the physical starting pose)
#   joint_target_q     the setpoint the controller drives; seeding it to the
#                      same value stops the arm lurching on frame 0
#   joint_target_ke    proportional gain (stiffness) of the position servo
#   joint_target_kd    derivative gain (damping); too low and the arm rings
#   joint_target_mode  position and/or velocity servo mode
#   joint_effort_limit torque ceiling, mirroring Part 1's ARM_FORCE_LIMIT

# Builder targets use coordinates; gains/limits use DOFs. The helper supplied
# in newton_scene.py matches joint names in MJCF actuator order, not a prefix
# of the eventual model (which also contains free payload joints).
robot_mj = mujoco.MjModel.from_xml_path(str(robot_xml))
joints = robot_joint_indices(builder, robot_mj)
gripper = set(spec.gripper_ctrl_indices)
for actuator, joint in enumerate(joints):
    q = builder.joint_q_start[joint]
    dof = builder.joint_qd_start[joint]
    builder.joint_q[q] = float(spec.home_ctrl[actuator])
    builder.joint_target_q[q] = float(spec.home_ctrl[actuator])
    builder.joint_target_ke[dof] = spec.arm_target_ke
    builder.joint_target_kd[dof] = spec.arm_target_kd
    builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)
    if actuator in gripper:
        continue
    builder.joint_effort_limit[dof] = spec.arm_effort_limit
    # Newton 1.5 preserves the actuator's independent force range as well.
    # Mirror Part 1's arm-only boost; leave reBot and both grippers unchanged.
    if spec.arm_force_limit is not None:
        builder.custom_attributes["mujoco:actuator_forcerange"].values[actuator] = wp.vec2(
            -spec.arm_force_limit, spec.arm_force_limit,
        )
