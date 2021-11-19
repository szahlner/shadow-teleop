def get_action():
    action = [0.] * env.action_space.shape[0]

    # Get the most recent frame
    frame = controller.frame()

    # Only process valid frames
    if frame.is_valid:

        # Only support a single right hand
        right_hand_processed = False

        # Process hands
        for hand in frame.hands:

            # Break after first right hand is processed
            if right_hand_processed:
                break

            # Process valid right hand
            if hand.is_right and hand.is_valid and not right_hand_processed:

                # THUMB
                theta1, theta2, theta3, theta4, theta5 = leap_motion_utils.thumb_joint_rotations(hand)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3) and not np.isnan(
                        theta4) and not np.isnan(theta5):
                    action[15:] = [
                        np.clip(theta1, -1.047, 1.047),
                        np.clip(theta2, 0., 1.222),
                        np.clip(theta3, -0.209, 0.209),
                        np.clip(theta4, -0.524, 0.524),
                        np.clip(theta5, 0., 1.571)]

                # INDEX finger
                index_finger = hand.fingers.finger_type(Leap.Finger.TYPE_INDEX)[0]

                theta1, theta2, theta3, _ = leap_motion_utils.finger_joint_rotations(index_finger)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
                    action[2:5] = [
                        np.clip(theta1, -0.349, 0.349),
                        np.clip(theta2, 0., 1.571),
                        np.clip(theta3, 0., 1.571)]

                # MIDDLE finger
                middle_finger = hand.fingers.finger_type(Leap.Finger.TYPE_MIDDLE)[0]

                theta1, theta2, theta3, _ = leap_motion_utils.finger_joint_rotations(middle_finger)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
                    action[5:8] = [
                        np.clip(theta1, -0.349, 0.349),
                        np.clip(theta2, 0., 1.571),
                        np.clip(theta3, 0., 1.571)]

                # RING finger
                ring_finger = hand.fingers.finger_type(Leap.Finger.TYPE_RING)[0]

                theta1, theta2, theta3, _ = leap_motion_utils.finger_joint_rotations(ring_finger)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
                    action[8:11] = [
                        -np.clip(theta1, -0.349, 0.349),  # "-" = BUG ?
                        np.clip(theta2, 0., 1.571),
                        np.clip(theta3, 0., 1.571)]

                # LITTLE finger
                little_finger = hand.fingers.finger_type(Leap.Finger.TYPE_PINKY)[0]

                theta1, theta2, theta3, _ = leap_motion_utils.finger_joint_rotations(little_finger)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
                    action[11:15] = [
                        0., # Little Finger 5
                        -np.clip(theta1, -0.349, 0.349),  # "-" = BUG ?
                        np.clip(theta2, 0., 1.571),
                        np.clip(theta3, 0., 1.571)]

                # WRIST
                action[0:2] = [
                    -leap_motion_utils.joint_rotation_y(hand.arm.basis, hand.basis, -0.489, 0.140),
                    leap_motion_utils.joint_rotation_x(hand.arm.basis, hand.basis, -0.698, 0.489)]

                # Mark right hand as processed
                right_hand_processed = True

    action = np.array(action).copy()

    if action.shape[0] != env.action_space.shape[0]:
        action = [0.] * env.action_space.shape[0]
        return np.array(action).copy()

    return action


def normalize_actions(action):
    joint_limit_low = []
    joint_limit_high = []
    joints_movable = []

    for n in range(env.n_model_joints):
        joint_info = pybullet.getJointInfo(env.model_id, n)

        if joint_info[1] in MOVABLE_JOINTS:
            joint_limit_low.append(joint_info[8])
            joint_limit_high.append(joint_info[9])
            joints_movable.append(n)

    joint_limit_low = np.array(joint_limit_low)
    joint_limit_high = np.array(joint_limit_high)

    act_range = (joint_limit_high - joint_limit_low) / 2.
    act_center = (joint_limit_high + joint_limit_low) / 2.

    return (action - act_center) / act_range