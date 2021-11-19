import numpy as np
from gym_shadow_hand.lib.leap_motion import Leap


def joint_rotation_x(prev_basis, next_basis, rot_angle_min=None, rot_angle_max=None):
    """
    Compute the joint rotation (angle) around the x-axis between the given bases.

    Parameters
    ----------
    prev_basis : Leap.Matrix
        Basis (x, y, z, origin) of the previous basis.
    next_basis : Leap.Matrix
        Basis (x, y, z, origin) of the next basis.
    rot_angle_min : float, optional
        Minimum joint rotation (angle) limit.
    rot_angle_max : float, optional
        Maximum joint rotation (angle) limit.

    Returns
    -------
    float
        Joint rotation (angle) around x-axis.
    """

    # Compute the rotation angle and axis between the x-bases
    rot_angle = prev_basis.x_basis.angle_to(next_basis.x_basis)
    rot_axis = prev_basis.x_basis.cross(next_basis.x_basis)

    # Set up corresponding rotation matrix
    rot_mat = Leap.Matrix()
    rot_mat.set_rotation(rot_axis, rot_angle)

    # Rotate next_basis in prev_basis
    rot_dir = rot_mat.transform_direction(next_basis.z_basis)

    # z-bases are in one plane, ready to measure the joint rotation (angle)
    rot_joint = prev_basis.z_basis.angle_to(rot_dir)
    cross = rot_dir.cross(prev_basis.z_basis)

    # Determine sign
    if cross.dot(prev_basis.x_basis) < 0:
        rot_joint *= -1

    # Minimum limit
    if rot_angle_min is not None and rot_joint < rot_angle_min:
        rot_joint = rot_angle_min

    # Maximum limit
    if rot_angle_max is not None and rot_joint > rot_angle_max:
        rot_joint = rot_angle_max

    return rot_joint


def joint_rotation_y(prev_basis, next_basis, rot_angle_min=None, rot_angle_max=None):
    """
    Compute the joint rotation (angle) around the y-axis between the given bases.

    Parameters
    ----------
    prev_basis : Leap.Matrix
        Basis (x, y, z, origin) of the previous basis.
    next_basis : Leap.Matrix
        Basis (x, y, z, origin) of the next basis.
    rot_angle_min : float, optional
        Minimum joint rotation (angle) limit.
    rot_angle_max : float, optional
        Maximum joint rotation (angle) limit.

    Returns
    -------
    float
        Joint rotation (angle) around y-axis.
    """

    # Compute the rotation angle and axis between the y-bases
    rot_angle = prev_basis.y_basis.angle_to(next_basis.y_basis)
    rot_axis = prev_basis.y_basis.cross(next_basis.y_basis)

    # Set up corresponding rotation matrix
    rot_mat = Leap.Matrix()
    rot_mat.set_rotation(rot_axis, rot_angle)

    # Rotate next_basis in prev_basis
    rot_dir = rot_mat.transform_direction(next_basis.x_basis)

    # x-bases are in one plane, ready to measure the joint rotation (angle)
    rot_joint = prev_basis.x_basis.angle_to(rot_dir)
    cross = rot_dir.cross(prev_basis.x_basis)

    # Determine sign
    if cross.dot(prev_basis.y_basis) < 0:
        rot_joint *= -1

    # Minimum limit
    if rot_angle_min is not None and rot_joint < rot_angle_min:
        rot_joint = rot_angle_min

    # Maximum limit
    if rot_angle_max is not None and rot_joint > rot_angle_max:
        rot_joint = rot_angle_max

    return rot_joint


def joint_rotation_z(prev_basis, next_basis, rot_angle_min=None, rot_angle_max=None):
    """
    Compute the joint rotation (angle) around the z-axis between the given bases.

    Parameters
    ----------
    prev_basis : Leap.Matrix
        Basis (x, y, z, origin) of the previous basis.
    next_basis : Leap.Matrix
        Basis (x, y, z, origin) of the next basis.
    rot_angle_min : float, optional
        Minimum joint rotation (angle) limit.
    rot_angle_max : float, optional
        Maximum joint rotation (angle) limit.

    Returns
    -------
    float
        Joint rotation (angle) around z-axis.
    """

    # Compute the rotation angle and axis between the z-bases
    rot_angle = prev_basis.z_basis.angle_to(next_basis.z_basis)
    rot_axis = prev_basis.z_basis.cross(next_basis.z_basis)

    # Set up corresponding rotation matrix
    rot_mat = Leap.Matrix()
    rot_mat.set_rotation(rot_axis, rot_angle)

    # Rotate next_basis in prev_basis
    rot_dir = rot_mat.transform_direction(next_basis.y_basis)

    # y-bases are in one plane, ready to measure the joint rotation (angle)
    rot_joint = prev_basis.y_basis.angle_to(rot_dir)
    cross = rot_dir.cross(prev_basis.y_basis)

    # Determine sign
    if cross.dot(prev_basis.z_basis) < 0:
        rot_joint *= -1

    # Minimum limit
    if rot_angle_min is not None and rot_joint < rot_angle_min:
        rot_joint = rot_angle_min

    # Maximum limit
    if rot_angle_max is not None and rot_joint > rot_angle_max:
        rot_joint = rot_angle_max

    return rot_joint


def finger_joint_rotations(finger):
    """
    Compute the joint rotations (angles) of the given finger.
    This works for the index, middle, ring and little finger.

    Parameters
    ----------
    finger : Leap.Finger
        Leap.Finger object. Index, Middle, Ring or Little finger.

    Returns
    -------
    theta1 : float
        Joint rotation (angle) of theta1 in radiant.
    theta2 : float
        Joint rotation (angle) of theta2 in radiant.
    theta3 : float
        Joint rotation (angle) of theta3 in radiant.
    theta4 : float
        Joint rotation (angle) of theta4 in radiant.
    """

    # Bases
    metacarpal_basis = finger.bone(Leap.Bone.TYPE_METACARPAL).basis
    proximal_basis = finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
    intermediate_basis = finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis

    # Transformations
    metacarpal_transform = Leap.Matrix(metacarpal_basis.x_basis,
                                       metacarpal_basis.y_basis,
                                       metacarpal_basis.z_basis,
                                       finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint)
    metacarpal_transform = metacarpal_transform.rigid_inverse()

    proximal_transform = Leap.Matrix(proximal_basis.x_basis,
                                     proximal_basis.y_basis,
                                     proximal_basis.z_basis,
                                     finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint)
    proximal_transform = proximal_transform.rigid_inverse()

    intermediate_transform = Leap.Matrix(intermediate_basis.x_basis,
                                         intermediate_basis.y_basis,
                                         intermediate_basis.z_basis,
                                         finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint)
    intermediate_transform = intermediate_transform.rigid_inverse()

    # End position - "endeffector"
    end_pos_metacarpal = finger.bone(Leap.Bone.TYPE_PROXIMAL).next_joint

    # Transform end position in current system
    vec = metacarpal_transform.transform_point(end_pos_metacarpal)
    l2 = finger.bone(Leap.Bone.TYPE_PROXIMAL).length

    # From inverse kinematics (DH-convention)
    theta1 = -np.arctan(vec.x / vec.z)
    theta2 = -np.arcsin(vec.y / l2)

    # End position - "endeffector"
    end_pos_proximal = finger.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint

    # Transform end position in current system
    vec = proximal_transform.transform_point(end_pos_proximal)
    l3 = finger.bone(Leap.Bone.TYPE_INTERMEDIATE).length

    # From inverse kinematics (DH-convention)
    theta3 = -np.arcsin(vec.y / l3)

    # Transform end position in current system
    end_pos_intermediate = finger.bone(Leap.Bone.TYPE_DISTAL).next_joint

    # Transform end position in current system
    vec = intermediate_transform.transform_point(end_pos_intermediate)
    l4 = finger.bone(Leap.Bone.TYPE_DISTAL).length

    # From inverse kinematics (DH-convention)
    theta4 = -np.arcsin(vec.y / l4)

    # Coupled joints
    if theta4 > theta3:
        theta4 = theta3

    return theta1, theta2, theta3, theta4


def little_finger_5_joint_rotation(finger):
    """
    Compute the joint rotation (angle) of little finger 5.

    Parameters
    ----------
    finger : Leap.Finger
        Leap.Finger object. Little finger.

    Returns
    -------
    theta : float
        Joint rotation (angle) of theta in radiant.
    """

    # Basis
    metacarpal_basis = finger.bone(Leap.Bone.TYPE_METACARPAL).basis

    # Transformation
    metacarpal_transform = Leap.Matrix(metacarpal_basis.x_basis,
                                       metacarpal_basis.y_basis,
                                       metacarpal_basis.z_basis,
                                       finger.bone(Leap.Bone.TYPE_METACARPAL).prev_joint)
    metacarpal_transform = metacarpal_transform.rigid_inverse()

    end_pos_metacarpal = finger.bone(Leap.Bone.TYPE_METACARPAL).next_joint
    vec = metacarpal_transform.transform_point(end_pos_metacarpal)

    l1 = finger.bone(Leap.Bone.TYPE_METACARPAL).length

    theta = np.pi - np.arccos((np.cos(7 * np.pi / 36) * np.cos(7 * np.pi / 36) * l1 + vec.z) / (
            np.cos(11 * np.pi / 36) * np.sin(7 * np.pi / 36) * l1))

    return theta


def thumb_joint_rotations(hand):
    """
    Compute the joint rotations (angles) of the thumb.

    Parameters
    ----------
    hand : Leap.Hand
        Leap.Hand object. Thumb only.

    Returns
    -------
    theta1 : float
        Joint rotation (angle) of theta1 in radiant.
    theta2 : float
        Joint rotation (angle) of theta2 in radiant.
    theta3 : float
        Joint rotation (angle) of theta3 in radiant.
    theta4 : float
        Joint rotation (angle) of theta4 in radiant.
    theta5 : float
        Joint rotation (angle) of theta5 in radiant.
    """

    thumb = hand.fingers.finger_type(Leap.Finger.TYPE_THUMB)[0]
    metacarpal_basis = thumb.bone(Leap.Bone.TYPE_METACARPAL).basis

    rot_angle = joint_rotation_y(hand.basis, metacarpal_basis) - 45 * Leap.DEG_TO_RAD
    rot_m = Leap.Matrix()
    rot_m.set_rotation(hand.basis.y_basis, rot_angle)
    x_basis = rot_m.transform_direction(hand.basis.x_basis)
    z_basis = rot_m.transform_direction(hand.basis.z_basis)

    metacarpal_basis = Leap.Matrix(x_basis,
                                   hand.basis.y_basis,
                                   z_basis,
                                   thumb.bone(Leap.Bone.TYPE_METACARPAL).next_joint)
    metacarpal_transform = metacarpal_basis.rigid_inverse()

    # End position - "endeffector"
    end_pos_metacarpal = thumb.bone(Leap.Bone.TYPE_PROXIMAL).next_joint

    # Transform end position in current system
    vec = metacarpal_transform.transform_point(end_pos_metacarpal)
    l2 = thumb.bone(Leap.Bone.TYPE_PROXIMAL).length

    # From inverse kinematics (DH-convention)
    try:
        theta1 = -np.arctan(vec.x / vec.y)
        theta2 = np.pi - np.arccos(vec.z / l2)
    except ZeroDivisionError:
        theta1 = np.nan
        theta2 = np.nan

    # Compute new proximal basis
    rot_m.set_rotation(z_basis, -theta1)
    x_basis = rot_m.transform_direction(x_basis)
    y_basis = rot_m.transform_direction(hand.basis.y_basis)

    rot_m.set_rotation(x_basis, theta2)
    y_basis = rot_m.transform_direction(y_basis)
    z_basis = rot_m.transform_direction(z_basis)

    proximal_basis = Leap.Matrix(x_basis,
                                 y_basis,
                                 z_basis,
                                 thumb.bone(Leap.Bone.TYPE_PROXIMAL).next_joint)
    proximal_transform = proximal_basis.rigid_inverse()

    # End position - "endeffector"
    end_pos_proximal = thumb.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint

    # Transform end position in current system
    vec = proximal_transform.transform_point(end_pos_proximal)
    l4 = thumb.bone(Leap.Bone.TYPE_INTERMEDIATE).length

    # From inverse kinematics (DH-convention)
    try:
        theta3 = np.arctan(vec.y / vec.z)
        theta4 = np.arcsin(vec.x / l4)
    except ZeroDivisionError:
        theta3 = np.nan
        theta4 = np.nan

    # Compute new intermediate basis
    rot_m.set_rotation(x_basis, theta3)
    y_basis = rot_m.transform_direction(y_basis)
    z_basis = rot_m.transform_direction(z_basis)

    rot_m.set_rotation(y_basis, theta4)
    x_basis = rot_m.transform_direction(x_basis)
    z_basis = rot_m.transform_direction(z_basis)

    intermediate_basis = Leap.Matrix(x_basis,
                                     y_basis,
                                     z_basis,
                                     thumb.bone(Leap.Bone.TYPE_INTERMEDIATE).next_joint)
    intermediate_transform = intermediate_basis.rigid_inverse()

    # End position - "endeffector"
    end_pos_intermediate = thumb.bone(Leap.Bone.TYPE_DISTAL).next_joint

    # Transform end position in current system
    vec = intermediate_transform.transform_point(end_pos_intermediate)
    l5 = thumb.bone(Leap.Bone.TYPE_DISTAL).length

    # From inverse kinematics (DH-convention)
    try:
        theta5 = np.pi - np.arccos(vec.z / l5)
    except ZeroDivisionError:
        theta5 = np.nan

    return theta1, theta2, theta3, theta4, theta5
