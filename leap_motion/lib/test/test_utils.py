import os
import matplotlib.pyplot as plt
import numpy as np

from lib.leap_motion import Leap

TEST_DATA_DIRECTORY = os.path.join("test", "leap_motion_test_data")
TEST_DATA_RESULTS_DIRECTORY = os.path.join("test", "results")
TEST_DATA_FILE_RIGHT_HAND = "right_hand.npy"
TEST_DATA_FILE_RIGHT_HAND_SAMPLES = "right_hand_samples.npy"

# THUMB POSITION
right_hand_thumb_proximal_dh_position_accuracy_mean = 0.
right_hand_thumb_intermediate_dh_position_accuracy_mean = 0.
right_hand_thumb_distal_dh_position_accuracy_mean = 0.

right_hand_thumb_proximal_dh_position_accuracy_limits_mean = 0.
right_hand_thumb_intermediate_dh_position_accuracy_limits_mean = 0.
right_hand_thumb_distal_dh_position_accuracy_limits_mean = 0.

# THUMB DIRECTION
right_hand_thumb_proximal_dh_direction_x_accuracy_mean = 0.
right_hand_thumb_proximal_dh_direction_y_accuracy_mean = 0.
right_hand_thumb_proximal_dh_direction_z_accuracy_mean = 0.
right_hand_thumb_intermediate_dh_direction_x_accuracy_mean = 0.
right_hand_thumb_intermediate_dh_direction_y_accuracy_mean = 0.
right_hand_thumb_intermediate_dh_direction_z_accuracy_mean = 0.
right_hand_thumb_distal_dh_direction_x_accuracy_mean = 0.
right_hand_thumb_distal_dh_direction_y_accuracy_mean = 0.
right_hand_thumb_distal_dh_direction_z_accuracy_mean = 0.

right_hand_thumb_proximal_dh_direction_x_accuracy_limits_mean = 0.
right_hand_thumb_proximal_dh_direction_y_accuracy_limits_mean = 0.
right_hand_thumb_proximal_dh_direction_z_accuracy_limits_mean = 0.
right_hand_thumb_intermediate_dh_direction_x_accuracy_limits_mean = 0.
right_hand_thumb_intermediate_dh_direction_y_accuracy_limits_mean = 0.
right_hand_thumb_intermediate_dh_direction_z_accuracy_limits_mean = 0.
right_hand_thumb_distal_dh_direction_x_accuracy_limits_mean = 0.
right_hand_thumb_distal_dh_direction_y_accuracy_limits_mean = 0.
right_hand_thumb_distal_dh_direction_z_accuracy_limits_mean = 0.

# INDEX finger POSITION
right_hand_index_finger_proximal_position_accuracy_mean = 0.
right_hand_index_finger_intermediate_position_accuracy_mean = 0.
right_hand_index_finger_distal_position_accuracy_mean = 0.
right_hand_index_finger_proximal_dh_position_accuracy_mean = 0.
right_hand_index_finger_intermediate_dh_position_accuracy_mean = 0.
right_hand_index_finger_distal_dh_position_accuracy_mean = 0.

right_hand_index_finger_proximal_position_accuracy_limits_mean = 0.
right_hand_index_finger_intermediate_position_accuracy_limits_mean = 0.
right_hand_index_finger_distal_position_accuracy_limits_mean = 0.
right_hand_index_finger_proximal_dh_position_accuracy_limits_mean = 0.
right_hand_index_finger_intermediate_dh_position_accuracy_limits_mean = 0.
right_hand_index_finger_distal_dh_position_accuracy_limits_mean = 0.

# INDEX finger DIRECTION
right_hand_index_finger_proximal_direction_x_accuracy_mean = 0.
right_hand_index_finger_proximal_direction_y_accuracy_mean = 0.
right_hand_index_finger_proximal_direction_z_accuracy_mean = 0.
right_hand_index_finger_intermediate_direction_x_accuracy_mean = 0.
right_hand_index_finger_intermediate_direction_y_accuracy_mean = 0.
right_hand_index_finger_intermediate_direction_z_accuracy_mean = 0.
right_hand_index_finger_distal_direction_x_accuracy_mean = 0.
right_hand_index_finger_distal_direction_y_accuracy_mean = 0.
right_hand_index_finger_distal_direction_z_accuracy_mean = 0.
right_hand_index_finger_proximal_dh_direction_x_accuracy_mean = 0.
right_hand_index_finger_proximal_dh_direction_y_accuracy_mean = 0.
right_hand_index_finger_proximal_dh_direction_z_accuracy_mean = 0.
right_hand_index_finger_intermediate_dh_direction_x_accuracy_mean = 0.
right_hand_index_finger_intermediate_dh_direction_y_accuracy_mean = 0.
right_hand_index_finger_intermediate_dh_direction_z_accuracy_mean = 0.
right_hand_index_finger_distal_dh_direction_x_accuracy_mean = 0.
right_hand_index_finger_distal_dh_direction_y_accuracy_mean = 0.
right_hand_index_finger_distal_dh_direction_z_accuracy_mean = 0.

right_hand_index_finger_proximal_direction_x_accuracy_limits_mean = 0.
right_hand_index_finger_proximal_direction_y_accuracy_limits_mean = 0.
right_hand_index_finger_proximal_direction_z_accuracy_limits_mean = 0.
right_hand_index_finger_intermediate_direction_x_accuracy_limits_mean = 0.
right_hand_index_finger_intermediate_direction_y_accuracy_limits_mean = 0.
right_hand_index_finger_intermediate_direction_z_accuracy_limits_mean = 0.
right_hand_index_finger_distal_direction_x_accuracy_limits_mean = 0.
right_hand_index_finger_distal_direction_y_accuracy_limits_mean = 0.
right_hand_index_finger_distal_direction_z_accuracy_limits_mean = 0.
right_hand_index_finger_proximal_dh_direction_x_accuracy_limits_mean = 0.
right_hand_index_finger_proximal_dh_direction_y_accuracy_limits_mean = 0.
right_hand_index_finger_proximal_dh_direction_z_accuracy_limits_mean = 0.
right_hand_index_finger_intermediate_dh_direction_x_accuracy_limits_mean = 0.
right_hand_index_finger_intermediate_dh_direction_y_accuracy_limits_mean = 0.
right_hand_index_finger_intermediate_dh_direction_z_accuracy_limits_mean = 0.
right_hand_index_finger_distal_dh_direction_x_accuracy_limits_mean = 0.
right_hand_index_finger_distal_dh_direction_y_accuracy_limits_mean = 0.
right_hand_index_finger_distal_dh_direction_z_accuracy_limits_mean = 0.

# Create test data directory in case it does not exist
if not os.path.isdir(TEST_DATA_DIRECTORY):
    os.mkdir(TEST_DATA_DIRECTORY)

# Create test data result directory in case it does not exist
if not os.path.isdir(TEST_DATA_RESULTS_DIRECTORY):
    os.mkdir(TEST_DATA_RESULTS_DIRECTORY)


def save_hand(hand, num, is_right=True):
    """
    Save hand data collected by the Leap Motion Controller.

    Parameters
    ----------
    hand : Leap.Hand
        Hand object to be saved.
    num : int
        How many samples.
    is_right : bool, optional
        Whether hand is right or left.

    Note
    ----
        The collected test data is less than a normal Leap.Hand object. It only contains necessary information.
    """

    # Set up test data dictionary
    # Only collect necessary information
    cur_data = dict()

    # HAND
    # Basis only
    cur_data["hand"] = {
        "basis": np.array([
            [hand.basis.x_basis.x, hand.basis.x_basis.y, hand.basis.x_basis.z],
            [hand.basis.y_basis.x, hand.basis.y_basis.y, hand.basis.y_basis.z],
            [hand.basis.z_basis.x, hand.basis.z_basis.y, hand.basis.z_basis.z]]),
        "palm_position": np.array([hand.palm_position.x,
                                   hand.palm_position.y,
                                   hand.palm_position.z]),
    }

    # ARM
    # Basis only
    cur_data["arm"] = {
        "basis": np.array([
            [hand.arm.basis.x_basis.x, hand.arm.basis.x_basis.y, hand.arm.basis.x_basis.z],
            [hand.arm.basis.y_basis.x, hand.arm.basis.y_basis.y, hand.arm.basis.y_basis.z],
            [hand.arm.basis.z_basis.x, hand.arm.basis.z_basis.y, hand.arm.basis.z_basis.z]]),
        "wrist_position": np.array([hand.arm.wrist_position.x,
                                    hand.arm.wrist_position.y,
                                    hand.arm.wrist_position.z]),
    }

    # THUMB
    # Get the Finger Object and the metacarpal basis
    thumb = hand.fingers.finger_type(Leap.Finger.TYPE_THUMB)[0]
    metacarpal_basis = thumb.bone(Leap.Bone.TYPE_METACARPAL).basis
    proximal_basis = thumb.bone(Leap.Bone.TYPE_PROXIMAL).basis
    intermediate_basis = thumb.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
    distal_basis = thumb.bone(Leap.Bone.TYPE_DISTAL).basis

    cur_data["thumb"] = {
        "metacarpal": {
            "basis": np.array([
                [metacarpal_basis.x_basis.x, metacarpal_basis.x_basis.y, metacarpal_basis.x_basis.z],
                [metacarpal_basis.y_basis.x, metacarpal_basis.y_basis.y, metacarpal_basis.y_basis.z],
                [metacarpal_basis.z_basis.x, metacarpal_basis.z_basis.y, metacarpal_basis.z_basis.z]]),
            "next_joint": np.array([thumb.bone(Leap.Bone.TYPE_METACARPAL).next_joint.x,
                                    thumb.bone(Leap.Bone.TYPE_METACARPAL).next_joint.y,
                                    thumb.bone(Leap.Bone.TYPE_METACARPAL).next_joint.z])
        },
        "proximal": {
            "basis": np.array([
                [proximal_basis.x_basis.x, proximal_basis.x_basis.y, proximal_basis.x_basis.z],
                [proximal_basis.y_basis.x, proximal_basis.y_basis.y, proximal_basis.y_basis.z],
                [proximal_basis.z_basis.x, proximal_basis.z_basis.y, proximal_basis.z_basis.z]]),
            "next_joint": np.array([thumb.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.x,
                                    thumb.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.y,
                                    thumb.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.z]),
            "length": thumb.bone(Leap.Bone.TYPE_PROXIMAL).length
        },
        "intermediate": {
            "basis": np.array([
                [intermediate_basis.x_basis.x, intermediate_basis.x_basis.y, intermediate_basis.x_basis.z],
                [intermediate_basis.y_basis.x, intermediate_basis.y_basis.y, intermediate_basis.y_basis.z],
                [intermediate_basis.z_basis.x, intermediate_basis.z_basis.y, intermediate_basis.z_basis.z]]),
            "next_joint": np.array([thumb.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.x,
                                    thumb.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.y,
                                    thumb.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.z]),
            "length": thumb.bone(Leap.Bone.TYPE_INTERMEDIATE).length
        },
        "distal": {
            "basis": np.array([
                [distal_basis.x_basis.x, distal_basis.x_basis.y, distal_basis.x_basis.z],
                [distal_basis.y_basis.x, distal_basis.y_basis.y, distal_basis.y_basis.z],
                [distal_basis.z_basis.x, distal_basis.z_basis.y, distal_basis.z_basis.z]]),
            "next_joint": np.array([thumb.bone(Leap.Bone.TYPE_DISTAL).next_joint.x,
                                    thumb.bone(Leap.Bone.TYPE_DISTAL).next_joint.y,
                                    thumb.bone(Leap.Bone.TYPE_DISTAL).next_joint.z]),
            "length": thumb.bone(Leap.Bone.TYPE_DISTAL).length
        }
    }

    # INDEX finger
    # Get the Finger Object and the bases of each bone in this finger
    index_finger = hand.fingers.finger_type(Leap.Finger.TYPE_INDEX)[0]
    metacarpal_basis = index_finger.bone(Leap.Bone.TYPE_METACARPAL).basis
    proximal_basis = index_finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
    intermediate_basis = index_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
    distal_basis = index_finger.bone(Leap.Bone.TYPE_DISTAL).basis

    cur_data["index_finger"] = {
        "metacarpal": {
            "basis": np.array([
                [metacarpal_basis.x_basis.x, metacarpal_basis.x_basis.y, metacarpal_basis.x_basis.z],
                [metacarpal_basis.y_basis.x, metacarpal_basis.y_basis.y, metacarpal_basis.y_basis.z],
                [metacarpal_basis.z_basis.x, metacarpal_basis.z_basis.y, metacarpal_basis.z_basis.z]]),
            "next_joint": np.array([index_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.x,
                                    index_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.y,
                                    index_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.z]),
            "length": index_finger.bone(Leap.Bone.TYPE_METACARPAL).length
        },
        "proximal": {
            "basis": np.array([
                [proximal_basis.x_basis.x, proximal_basis.x_basis.y, proximal_basis.x_basis.z],
                [proximal_basis.y_basis.x, proximal_basis.y_basis.y, proximal_basis.y_basis.z],
                [proximal_basis.z_basis.x, proximal_basis.z_basis.y, proximal_basis.z_basis.z]]),
            "next_joint": np.array([index_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.x,
                                    index_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.y,
                                    index_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.z]),
            "length": index_finger.bone(Leap.Bone.TYPE_PROXIMAL).length
        },
        "intermediate": {
            "basis": np.array([
                [intermediate_basis.x_basis.x, intermediate_basis.x_basis.y, intermediate_basis.x_basis.z],
                [intermediate_basis.y_basis.x, intermediate_basis.y_basis.y, intermediate_basis.y_basis.z],
                [intermediate_basis.z_basis.x, intermediate_basis.z_basis.y, intermediate_basis.z_basis.z]]),
            "next_joint": np.array([index_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.x,
                                    index_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.y,
                                    index_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.z]),
            "length": index_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).length
        },
        "distal": {
            "basis": np.array([
                [distal_basis.x_basis.x, distal_basis.x_basis.y, distal_basis.x_basis.z],
                [distal_basis.y_basis.x, distal_basis.y_basis.y, distal_basis.y_basis.z],
                [distal_basis.z_basis.x, distal_basis.z_basis.y, distal_basis.z_basis.z]]),
            "next_joint": np.array([index_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.x,
                                    index_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.y,
                                    index_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.z]),
            "length": index_finger.bone(Leap.Bone.TYPE_DISTAL).length
        }
    }

    # MIDDLE finger
    # Get the Finger Object and the bases of each bone in this finger
    middle_finger = hand.fingers.finger_type(Leap.Finger.TYPE_MIDDLE)[0]
    metacarpal_basis = middle_finger.bone(Leap.Bone.TYPE_METACARPAL).basis
    proximal_basis = middle_finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
    intermediate_basis = middle_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
    distal_basis = middle_finger.bone(Leap.Bone.TYPE_DISTAL).basis

    cur_data["middle_finger"] = {
        "metacarpal": {
            "basis": np.array([
                [metacarpal_basis.x_basis.x, metacarpal_basis.x_basis.y, metacarpal_basis.x_basis.z],
                [metacarpal_basis.y_basis.x, metacarpal_basis.y_basis.y, metacarpal_basis.y_basis.z],
                [metacarpal_basis.z_basis.x, metacarpal_basis.z_basis.y, metacarpal_basis.z_basis.z]]),
            "next_joint": np.array([middle_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.x,
                                    middle_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.y,
                                    middle_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.z]),
            "length": middle_finger.bone(Leap.Bone.TYPE_METACARPAL).length
        },
        "proximal": {
            "basis": np.array([
                [proximal_basis.x_basis.x, proximal_basis.x_basis.y, proximal_basis.x_basis.z],
                [proximal_basis.y_basis.x, proximal_basis.y_basis.y, proximal_basis.y_basis.z],
                [proximal_basis.z_basis.x, proximal_basis.z_basis.y, proximal_basis.z_basis.z]]),
            "next_joint": np.array([middle_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.x,
                                    middle_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.y,
                                    middle_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.z]),
            "length": middle_finger.bone(Leap.Bone.TYPE_PROXIMAL).length
        },
        "intermediate": {
            "basis": np.array([
                [intermediate_basis.x_basis.x, intermediate_basis.x_basis.y, intermediate_basis.x_basis.z],
                [intermediate_basis.y_basis.x, intermediate_basis.y_basis.y, intermediate_basis.y_basis.z],
                [intermediate_basis.z_basis.x, intermediate_basis.z_basis.y, intermediate_basis.z_basis.z]]),
            "next_joint": np.array([middle_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.x,
                                    middle_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.y,
                                    middle_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.z]),
            "length": middle_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).length
        },
        "distal": {
            "basis": np.array([
                [distal_basis.x_basis.x, distal_basis.x_basis.y, distal_basis.x_basis.z],
                [distal_basis.y_basis.x, distal_basis.y_basis.y, distal_basis.y_basis.z],
                [distal_basis.z_basis.x, distal_basis.z_basis.y, distal_basis.z_basis.z]]),
            "next_joint": np.array([middle_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.x,
                                    middle_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.y,
                                    middle_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.z]),
            "length": middle_finger.bone(Leap.Bone.TYPE_DISTAL).length
        }
    }

    # RING finger
    # Get the Finger Object and the bases of each bone in this finger
    ring_finger = hand.fingers.finger_type(Leap.Finger.TYPE_RING)[0]
    metacarpal_basis = ring_finger.bone(Leap.Bone.TYPE_METACARPAL).basis
    proximal_basis = ring_finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
    intermediate_basis = ring_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
    distal_basis = ring_finger.bone(Leap.Bone.TYPE_DISTAL).basis

    cur_data["ring_finger"] = {
        "metacarpal": {
            "basis": np.array([
                [metacarpal_basis.x_basis.x, metacarpal_basis.x_basis.y, metacarpal_basis.x_basis.z],
                [metacarpal_basis.y_basis.x, metacarpal_basis.y_basis.y, metacarpal_basis.y_basis.z],
                [metacarpal_basis.z_basis.x, metacarpal_basis.z_basis.y, metacarpal_basis.z_basis.z]]),
            "next_joint": np.array([ring_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.x,
                                    ring_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.y,
                                    ring_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.z]),
            "length": ring_finger.bone(Leap.Bone.TYPE_METACARPAL).length
        },
        "proximal": {
            "basis": np.array([
                [proximal_basis.x_basis.x, proximal_basis.x_basis.y, proximal_basis.x_basis.z],
                [proximal_basis.y_basis.x, proximal_basis.y_basis.y, proximal_basis.y_basis.z],
                [proximal_basis.z_basis.x, proximal_basis.z_basis.y, proximal_basis.z_basis.z]]),
            "next_joint": np.array([ring_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.x,
                                    ring_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.y,
                                    ring_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.z]),
            "length": ring_finger.bone(Leap.Bone.TYPE_PROXIMAL).length
        },
        "intermediate": {
            "basis": np.array([
                [intermediate_basis.x_basis.x, intermediate_basis.x_basis.y, intermediate_basis.x_basis.z],
                [intermediate_basis.y_basis.x, intermediate_basis.y_basis.y, intermediate_basis.y_basis.z],
                [intermediate_basis.z_basis.x, intermediate_basis.z_basis.y, intermediate_basis.z_basis.z]]),
            "next_joint": np.array([ring_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.x,
                                    ring_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.y,
                                    ring_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.z]),
            "length": ring_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).length
        },
        "distal": {
            "basis": np.array([
                [distal_basis.x_basis.x, distal_basis.x_basis.y, distal_basis.x_basis.z],
                [distal_basis.y_basis.x, distal_basis.y_basis.y, distal_basis.y_basis.z],
                [distal_basis.z_basis.x, distal_basis.z_basis.y, distal_basis.z_basis.z]]),
            "next_joint": np.array([ring_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.x,
                                    ring_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.y,
                                    ring_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.z]),
            "length": ring_finger.bone(Leap.Bone.TYPE_DISTAL).length
        }
    }

    # LITTLE finger
    # Get the Finger Object and the bases of each bone in this finger
    little_finger = hand.fingers.finger_type(Leap.Finger.TYPE_PINKY)[0]
    metacarpal_basis = little_finger.bone(Leap.Bone.TYPE_METACARPAL).basis
    proximal_basis = little_finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
    intermediate_basis = little_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
    distal_basis = little_finger.bone(Leap.Bone.TYPE_DISTAL).basis

    cur_data["little_finger"] = {
        "metacarpal": {
            "basis": np.array([
                [metacarpal_basis.x_basis.x, metacarpal_basis.x_basis.y, metacarpal_basis.x_basis.z],
                [metacarpal_basis.y_basis.x, metacarpal_basis.y_basis.y, metacarpal_basis.y_basis.z],
                [metacarpal_basis.z_basis.x, metacarpal_basis.z_basis.y, metacarpal_basis.z_basis.z]]),
            "prev_joint": np.array([little_finger.bone(Leap.Bone.TYPE_METACARPAL).prev_joint.x,
                                    little_finger.bone(Leap.Bone.TYPE_METACARPAL).prev_joint.y,
                                    little_finger.bone(Leap.Bone.TYPE_METACARPAL).prev_joint.z]),
            "next_joint": np.array([little_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.x,
                                    little_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.y,
                                    little_finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint.z]),
            "length": little_finger.bone(Leap.Bone.TYPE_METACARPAL).length
        },
        "proximal": {
            "basis": np.array([
                [proximal_basis.x_basis.x, proximal_basis.x_basis.y, proximal_basis.x_basis.z],
                [proximal_basis.y_basis.x, proximal_basis.y_basis.y, proximal_basis.y_basis.z],
                [proximal_basis.z_basis.x, proximal_basis.z_basis.y, proximal_basis.z_basis.z]]),
            "next_joint": np.array([little_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.x,
                                    little_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.y,
                                    little_finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint.z]),
            "length": little_finger.bone(Leap.Bone.TYPE_PROXIMAL).length
        },
        "intermediate": {
            "basis": np.array([
                [intermediate_basis.x_basis.x, intermediate_basis.x_basis.y, intermediate_basis.x_basis.z],
                [intermediate_basis.y_basis.x, intermediate_basis.y_basis.y, intermediate_basis.y_basis.z],
                [intermediate_basis.z_basis.x, intermediate_basis.z_basis.y, intermediate_basis.z_basis.z]]),
            "next_joint": np.array([little_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.x,
                                    little_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.y,
                                    little_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint.z]),
            "length": little_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).length
        },
        "distal": {
            "basis": np.array([
                [distal_basis.x_basis.x, distal_basis.x_basis.y, distal_basis.x_basis.z],
                [distal_basis.y_basis.x, distal_basis.y_basis.y, distal_basis.y_basis.z],
                [distal_basis.z_basis.x, distal_basis.z_basis.y, distal_basis.z_basis.z]]),
            "next_joint": np.array([little_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.x,
                                    little_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.y,
                                    little_finger.bone(Leap.Bone.TYPE_DISTAL).next_joint.z]),
            "length": little_finger.bone(Leap.Bone.TYPE_DISTAL).length
        }
    }

    # Set up filename and save the collected data samples (count)
    file_name_samples = os.path.join(TEST_DATA_DIRECTORY, TEST_DATA_FILE_RIGHT_HAND_SAMPLES)
    with open(file_name_samples, "wb") as f:
        np.save(f, num)
        f.close()

    file_name = os.path.join(TEST_DATA_DIRECTORY, TEST_DATA_FILE_RIGHT_HAND)
    with open(file_name, "ab") as f:
        np.save(f, cur_data)
        f.close()


def load_hand(is_right=True):
    """
    Load the collected hand data.

    Parameters
    ----------
    is_right : bool, optional
        Whether hand is right or left.

    Returns
    -------
    dict
        Test data in a dictionary.

    Note
    ----
        The collected test data is less than a normal Leap.Hand object. It only contains necessary information.
    """

    num = get_num_samples(is_right=is_right)

    file_name = os.path.join(TEST_DATA_DIRECTORY, TEST_DATA_FILE_RIGHT_HAND)
    with open(file_name, "rb") as f:
        test_data = np.stack([np.load(f, allow_pickle=True) for _ in range(num)])
        f.close()

    return test_data


def get_num_samples(is_right=True):
    """
    Gets number of data samples.

    Parameters
    ----------
    is_right : bool, optional
        Whether hand is right or left.

    Returns
    -------
    int
        Number of data samples.
    """

    file_name = os.path.join(TEST_DATA_DIRECTORY, TEST_DATA_FILE_RIGHT_HAND_SAMPLES)

    try:
        with open(file_name, "rb") as f:
            num = np.load(f)
            num = num.item()
            f.close()
    except FileNotFoundError:
        num = 0

    return num


def clear_samples():
    """
    Clears all sample data.
    """

    file_name = os.path.join(TEST_DATA_DIRECTORY, TEST_DATA_FILE_RIGHT_HAND)
    if os.path.isfile(file_name):
        os.remove(file_name)

    file_name = os.path.join(TEST_DATA_DIRECTORY, TEST_DATA_FILE_RIGHT_HAND_SAMPLES)
    if os.path.isfile(file_name):
        os.remove(file_name)


def plot_stats():
    """Combines all figures."""

    if TEST_NUMBER_DATA_SAMPLES == 0:
        return

    with plt.style.context('ggplot'):
        face_color = [(0., 0.4470, 0.7410, 1.),
                      (0.3010, 0.7450, 0.9330, 1.),
                      (0.8500, 0.3250, 0.0980, 1.),
                      (0.9290, 0.6940, 0.1250, 1.)]
        face_color_thumb = [(0., 0.4470, 0.7410, 1.), (0.3010, 0.7450, 0.9330, 1.)]

        labels = ["N", "DH", "N-L", "DH-L"]
        labels_thumb = ["DH", "DH-L"]

        # Plot THUMB POSITION
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), constrained_layout=True)

        axes[0].bar(x=[1, 2],
                    height=[right_hand_thumb_proximal_dh_position_accuracy_mean,
                            right_hand_thumb_proximal_dh_position_accuracy_limits_mean],
                    color=face_color_thumb, tick_label=labels_thumb)
        axes[0].set_xticklabels(labels)
        axes[0].set_yscale('log')
        axes[0].set_ylabel("Quantity")
        axes[0].set_title("Proximal", fontsize=12)

        axes[1].bar(x=[1, 2],
                    height=[right_hand_thumb_intermediate_dh_position_accuracy_mean,
                            right_hand_thumb_intermediate_dh_position_accuracy_limits_mean],
                    color=face_color_thumb, tick_label=labels_thumb)
        axes[1].set_yscale('log')
        axes[1].set_xlabel("Distance error in mm", labelpad=20)
        axes[1].set_title("Intermediate", fontsize=12)

        axes[2].bar(x=[1, 2],
                    height=[right_hand_thumb_distal_dh_position_accuracy_mean,
                            right_hand_thumb_distal_dh_position_accuracy_limits_mean],
                    color=face_color_thumb, tick_label=labels_thumb)
        axes[2].set_yscale('log')
        axes[2].set_title("Distal", fontsize=12)

        fig.suptitle("Right Hand Thumb Bone End-Position Mean-Error, samples: {}".format(TEST_NUMBER_DATA_SAMPLES), fontsize=14)

        fig.savefig(os.path.join(TEST_DATA_RESULTS_DIRECTORY, "right_hand_thumb_position_accuracy_mean.png"), format="png")
        plt.close()

        # Plot THUMB DIRECTION
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9), constrained_layout=True)

        axes[0][0].bar(x=[1, 2],
                       height=[right_hand_thumb_proximal_dh_direction_x_accuracy_mean,
                               right_hand_thumb_proximal_dh_direction_x_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[0][0].set_xticklabels(labels)
        axes[0][0].set_yscale('log')
        axes[0][0].set_ylabel("Quantity")
        axes[0][0].set_title("Proximal x", fontsize=12)

        axes[0][1].bar(x=[1, 2],
                       height=[right_hand_thumb_proximal_dh_direction_y_accuracy_mean,
                               right_hand_thumb_proximal_dh_direction_y_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[0][1].set_yscale('log')
        axes[0][1].set_title("Proximal y", fontsize=12)

        axes[0][2].bar(x=[1, 2],
                       height=[right_hand_thumb_proximal_dh_direction_z_accuracy_mean,
                               right_hand_thumb_proximal_dh_direction_z_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[0][2].set_yscale('log')
        axes[0][2].set_title("Proximal z", fontsize=12)

        axes[1][0].bar(x=[1, 2],
                       height=[right_hand_thumb_intermediate_dh_direction_x_accuracy_mean,
                               right_hand_thumb_intermediate_dh_direction_x_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[1][0].set_xticklabels(labels)
        axes[1][0].set_yscale('log')
        axes[1][0].set_ylabel("Quantity")
        axes[1][0].set_title("Intermediate x", fontsize=12)

        axes[1][1].bar(x=[1, 2],
                       height=[right_hand_thumb_intermediate_dh_direction_y_accuracy_mean,
                               right_hand_thumb_intermediate_dh_direction_y_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[1][1].set_yscale('log')
        axes[1][1].set_title("Intermediate y", fontsize=12)

        axes[1][2].bar(x=[1, 2],
                       height=[right_hand_thumb_intermediate_dh_direction_z_accuracy_mean,
                               right_hand_thumb_intermediate_dh_direction_z_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[1][2].set_yscale('log')
        axes[1][2].set_title("Intermediate z", fontsize=12)

        axes[2][0].bar(x=[1, 2],
                       height=[right_hand_thumb_distal_dh_direction_x_accuracy_mean,
                               right_hand_thumb_distal_dh_direction_x_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[2][0].set_xticklabels(labels)
        axes[2][0].set_yscale('log')
        axes[2][0].set_ylabel("Quantity")
        axes[2][0].set_title("Distal x", fontsize=12)

        axes[2][1].bar(x=[1, 2],
                       height=[right_hand_thumb_distal_dh_direction_y_accuracy_mean,
                               right_hand_thumb_distal_dh_direction_y_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[2][1].set_yscale('log')
        axes[2][1].set_xlabel("Distance error in mm", labelpad=20)
        axes[2][1].set_title("Distal y", fontsize=12)

        axes[2][2].bar(x=[1, 2],
                       height=[right_hand_thumb_distal_dh_direction_z_accuracy_mean,
                               right_hand_thumb_distal_dh_direction_z_accuracy_limits_mean],
                       color=face_color_thumb, tick_label=labels_thumb)
        axes[2][2].set_yscale('log')
        axes[2][2].set_title("Distal z", fontsize=12)

        fig.suptitle("Right Hand Thumb Bone Direction Mean-Error, samples: {}".format(TEST_NUMBER_DATA_SAMPLES), fontsize=14)

        fig.savefig(os.path.join(TEST_DATA_RESULTS_DIRECTORY, "right_hand_thumb_direction_accuracy_mean.png"), format="png")
        plt.close()

        # Plot INDEX finger POSITION
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), constrained_layout=True)

        axes[0].bar(x=[1, 2, 3, 4],
                    height=[right_hand_index_finger_proximal_position_accuracy_mean,
                            right_hand_index_finger_proximal_dh_position_accuracy_mean,
                            right_hand_index_finger_proximal_position_accuracy_limits_mean,
                            right_hand_index_finger_proximal_dh_position_accuracy_limits_mean],
                    color=face_color, tick_label=labels)
        axes[0].set_xticklabels(labels)
        axes[0].set_yscale('log')
        axes[0].set_ylabel("Quantity")
        axes[0].set_title("Proximal", fontsize=12)

        axes[1].bar(x=[1, 2, 3, 4],
                    height=[right_hand_index_finger_intermediate_position_accuracy_mean,
                            right_hand_index_finger_intermediate_dh_position_accuracy_mean,
                            right_hand_index_finger_intermediate_position_accuracy_limits_mean,
                            right_hand_index_finger_intermediate_dh_position_accuracy_limits_mean],
                    color=face_color, tick_label=labels)
        axes[1].set_yscale('log')
        axes[1].set_xlabel("Distance error in mm", labelpad=20)
        axes[1].set_title("Intermediate", fontsize=12)

        axes[2].bar(x=[1, 2, 3, 4],
                    height=[right_hand_index_finger_distal_position_accuracy_mean,
                            right_hand_index_finger_distal_dh_position_accuracy_mean,
                            right_hand_index_finger_distal_position_accuracy_limits_mean,
                            right_hand_index_finger_distal_dh_position_accuracy_limits_mean],
                    color=face_color, tick_label=labels)
        axes[2].set_yscale('log')
        axes[2].set_title("Distal", fontsize=12)

        fig.suptitle("Right Hand Index Finger Bone End-Position Mean-Error, samples: {}".format(TEST_NUMBER_DATA_SAMPLES), fontsize=14)

        fig.savefig(os.path.join(TEST_DATA_RESULTS_DIRECTORY, "right_hand_index_finger_position_accuracy_mean.png"), format="png")
        plt.close()

        # Plot INDEX finger DIRECTION
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9), constrained_layout=True)

        axes[0][0].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_proximal_direction_x_accuracy_mean,
                               right_hand_index_finger_proximal_dh_direction_x_accuracy_mean,
                               right_hand_index_finger_proximal_direction_x_accuracy_limits_mean,
                               right_hand_index_finger_proximal_dh_direction_x_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[0][0].set_xticklabels(labels)
        axes[0][0].set_yscale('log')
        axes[0][0].set_ylabel("Quantity")
        axes[0][0].set_title("Proximal x", fontsize=12)

        axes[0][1].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_proximal_direction_y_accuracy_mean,
                               right_hand_index_finger_proximal_dh_direction_y_accuracy_mean,
                               right_hand_index_finger_proximal_direction_y_accuracy_limits_mean,
                               right_hand_index_finger_proximal_dh_direction_y_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[0][1].set_yscale('log')
        axes[0][1].set_title("Proximal y", fontsize=12)

        axes[0][2].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_proximal_direction_z_accuracy_mean,
                               right_hand_index_finger_proximal_dh_direction_z_accuracy_mean,
                               right_hand_index_finger_proximal_direction_z_accuracy_limits_mean,
                               right_hand_index_finger_proximal_dh_direction_z_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[0][2].set_yscale('log')
        axes[0][2].set_title("Proximal z", fontsize=12)

        axes[1][0].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_intermediate_direction_x_accuracy_mean,
                               right_hand_index_finger_intermediate_dh_direction_x_accuracy_mean,
                               right_hand_index_finger_intermediate_direction_x_accuracy_limits_mean,
                               right_hand_index_finger_intermediate_dh_direction_x_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[1][0].set_xticklabels(labels)
        axes[1][0].set_yscale('log')
        axes[1][0].set_ylabel("Quantity")
        axes[1][0].set_title("Intermediate x", fontsize=12)

        axes[1][1].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_intermediate_direction_y_accuracy_mean,
                               right_hand_index_finger_intermediate_dh_direction_y_accuracy_mean,
                               right_hand_index_finger_intermediate_direction_y_accuracy_limits_mean,
                               right_hand_index_finger_intermediate_dh_direction_y_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[1][1].set_yscale('log')
        axes[1][1].set_title("Intermediate y", fontsize=12)

        axes[1][2].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_intermediate_direction_z_accuracy_mean,
                               right_hand_index_finger_intermediate_dh_direction_z_accuracy_mean,
                               right_hand_index_finger_intermediate_direction_z_accuracy_limits_mean,
                               right_hand_index_finger_intermediate_dh_direction_z_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[1][2].set_yscale('log')
        axes[1][2].set_title("Intermediate z", fontsize=12)

        axes[2][0].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_distal_direction_x_accuracy_mean,
                               right_hand_index_finger_distal_dh_direction_x_accuracy_mean,
                               right_hand_index_finger_distal_direction_x_accuracy_limits_mean,
                               right_hand_index_finger_distal_dh_direction_x_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[2][0].set_xticklabels(labels)
        axes[2][0].set_yscale('log')
        axes[2][0].set_ylabel("Quantity")
        axes[2][0].set_title("Distal x", fontsize=12)

        axes[2][1].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_distal_direction_y_accuracy_mean,
                               right_hand_index_finger_distal_dh_direction_y_accuracy_mean,
                               right_hand_index_finger_distal_direction_y_accuracy_limits_mean,
                               right_hand_index_finger_distal_dh_direction_y_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[2][1].set_yscale('log')
        axes[2][1].set_xlabel("Distance error in mm", labelpad=20)
        axes[2][1].set_title("Distal y", fontsize=12)

        axes[2][2].bar(x=[1, 2, 3, 4],
                       height=[right_hand_index_finger_distal_direction_z_accuracy_mean,
                               right_hand_index_finger_distal_dh_direction_z_accuracy_mean,
                               right_hand_index_finger_distal_direction_z_accuracy_limits_mean,
                               right_hand_index_finger_distal_dh_direction_z_accuracy_limits_mean],
                       color=face_color, tick_label=labels)
        axes[2][2].set_yscale('log')
        axes[2][2].set_title("Distal z", fontsize=12)

        fig.suptitle("Right Hand Index Finger Bone Direction Mean-Error, samples: {}".format(TEST_NUMBER_DATA_SAMPLES), fontsize=14)

        fig.savefig(os.path.join(TEST_DATA_RESULTS_DIRECTORY, "right_hand_index_finger_direction_accuracy_mean.png"), format="png")
        plt.close()


TEST_NUMBER_DATA_SAMPLES = get_num_samples(is_right=True)
