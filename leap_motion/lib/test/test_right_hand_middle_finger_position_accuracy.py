import os
import unittest
from parameterized import parameterized
import matplotlib.pyplot as plt
import numpy as np

from lib.leap_motion import Leap
from lib.test import test_utils
from lib.test.argument_parser import wrapper
from lib import utils


class TestRightHandMiddleFingerPositionAccuracy(unittest.TestCase):
    right_hand_middle_finger_proximal_position_accuracy = []
    right_hand_middle_finger_intermediate_position_accuracy = []
    right_hand_middle_finger_distal_position_accuracy = []

    right_hand_middle_finger_proximal_dh_position_accuracy = []
    right_hand_middle_finger_intermediate_dh_position_accuracy = []
    right_hand_middle_finger_distal_dh_position_accuracy = []

    @classmethod
    def tearDownClass(cls):
        if test_utils.TEST_NUMBER_DATA_SAMPLES == 0:
            return

        with plt.style.context('ggplot'):
            face_color = (0., 0.4470, 0.7410, 1.)
            
            # Plot 1
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), constrained_layout=True)

            axes[0].hist(cls.right_hand_middle_finger_proximal_position_accuracy, bins=50, facecolor=face_color)
            axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[0].set_ylabel("Quantity")
            axes[0].set_title("Proximal", fontsize=12)

            axes[1].hist(cls.right_hand_middle_finger_intermediate_position_accuracy, bins=50, facecolor=face_color)
            axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[1].set_xlabel("Distance error in mm", labelpad=20)
            axes[1].set_title("Intermediate", fontsize=12)

            axes[2].hist(cls.right_hand_middle_finger_distal_position_accuracy, bins=50, facecolor=face_color)
            axes[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[2].set_title("Distal", fontsize=12)

            fig.suptitle("Right Hand Middle Finger Bone End-Position, samples: {}".format(len(cls.right_hand_middle_finger_proximal_position_accuracy)), fontsize=14)

            fig.savefig(os.path.join(test_utils.TEST_DATA_RESULTS_DIRECTORY, "right_hand_middle_finger_position_accuracy.png"), format="png")
            plt.close()

            face_color = (0.8500, 0.3250, 0.0980, 1.)

            # Plot 2
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), constrained_layout=True)

            axes[0].hist(cls.right_hand_middle_finger_proximal_dh_position_accuracy, bins=50, facecolor=face_color)
            axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[0].set_ylabel("Quantity")
            axes[0].set_title("Proximal", fontsize=12)

            axes[1].hist(cls.right_hand_middle_finger_intermediate_dh_position_accuracy, bins=50, facecolor=face_color)
            axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[1].set_xlabel("Distance error in mm", labelpad=20)
            axes[1].set_title("Intermediate", fontsize=12)

            axes[2].hist(cls.right_hand_middle_finger_distal_dh_position_accuracy, bins=50, facecolor=face_color)
            axes[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[2].set_title("Distal", fontsize=12)

            fig.suptitle("Right Hand Middle Finger DH Bone End-Position, samples: {}".format(len(cls.right_hand_middle_finger_proximal_dh_position_accuracy)), fontsize=14)

            fig.savefig(os.path.join(test_utils.TEST_DATA_RESULTS_DIRECTORY, "right_hand_middle_finger_dh_position_accuracy.png"), format="png")
            plt.close()

    @classmethod
    def setUpClass(cls):
        if test_utils.TEST_NUMBER_DATA_SAMPLES != 0:
            cls.hand = test_utils.load_hand()

    def setUp(self):
        if test_utils.TEST_NUMBER_DATA_SAMPLES == 0:
            self.skipTest("no sample data available")

    @parameterized.expand([
        ["right_hand_middle_finger_proximal_position_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger positions, proximal bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone top is compared to the real position and the difference is
        compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.
        """

        # Prepare bases
        basis = self.hand[num]["middle_finger"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["proximal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        proximal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare positions
        vec = self.hand[num]["middle_finger"]["proximal"]["next_joint"]
        real_pos = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["metacarpal"]["next_joint"]
        cur_pos = Leap.Vector(vec[0], vec[1], vec[2])

        # Construct the finger from the metacarpal bone top (next_joint)
        rot_mat = Leap.Matrix()
        rot_x = utils.joint_rotation_x(metacarpal_basis, proximal_basis)
        rot_mat.set_rotation(metacarpal_basis.x_basis, rot_x)
        new_dir = rot_mat.transform_direction(metacarpal_basis.z_basis)

        rot_y = utils.joint_rotation_y(metacarpal_basis, proximal_basis)
        rot_mat.set_rotation(metacarpal_basis.y_basis, rot_y)
        new_dir = rot_mat.transform_direction(new_dir)

        # Add positions
        cur_pos += -new_dir * self.hand[num]["middle_finger"]["proximal"]["length"]

        # Compare
        diff = cur_pos.distance_to(real_pos)
        self.right_hand_middle_finger_proximal_position_accuracy.append(diff)

        self.assertAlmostEqual(diff, 0., places=accuracy)

    @parameterized.expand([
        ["right_hand_middle_finger_proximal_dh_position_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger positions, proximal bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone top is compared to the real position and the difference is
        compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.
        """

        # Prepare bases
        basis = self.hand[num]["middle_finger"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare positions
        vec = self.hand[num]["middle_finger"]["proximal"]["next_joint"]
        real_pos = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["metacarpal"]["next_joint"]
        cur_pos = Leap.Vector(vec[0], vec[1], vec[2])

        # Transformations
        metacarpal_transform = Leap.Matrix(metacarpal_basis.x_basis,
                                           metacarpal_basis.y_basis,
                                           metacarpal_basis.z_basis,
                                           cur_pos)
        metacarpal_transform = metacarpal_transform.rigid_inverse()

        # End position - "endeffector"
        end_pos_metacarpal = real_pos

        # Transform end position in current system
        vec = metacarpal_transform.transform_point(end_pos_metacarpal)
        l2 = self.hand[num]["middle_finger"]["proximal"]["length"]

        # From inverse kinematics (DH-convention)
        theta1 = -np.arctan(vec.x / vec.z)
        theta2 = -np.arcsin(vec.y / l2)

        # Construct the finger from the metacarpal bone top (next_joint)
        rot_theta1 = Leap.Matrix()
        rot_theta1.set_rotation(metacarpal_basis.y_basis, theta1)
        rot_theta2 = Leap.Matrix()
        rot_theta2.set_rotation(metacarpal_basis.x_basis, theta2)
        rot_m = rot_theta1 * rot_theta2
        new_dir = rot_m.transform_direction(metacarpal_basis.z_basis)

        # Add positions
        cur_pos += -new_dir * l2

        # Compare
        diff = cur_pos.distance_to(real_pos)
        self.right_hand_middle_finger_proximal_dh_position_accuracy.append(diff)

        self.assertAlmostEqual(diff, 0., places=accuracy)

    @parameterized.expand([
        ["right_hand_middle_finger_intermediate_position_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger positions, intermediate bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone top is compared to the real position and the difference is
        compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.
        """

        # Prepare bases
        basis = self.hand[num]["middle_finger"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["proximal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        proximal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["intermediate"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        intermediate_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare positions
        vec = self.hand[num]["middle_finger"]["intermediate"]["next_joint"]
        real_pos = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["metacarpal"]["next_joint"]
        cur_pos = Leap.Vector(vec[0], vec[1], vec[2])

        # Construct the finger from the metacarpal bone top (next_joint)
        rot_mat = Leap.Matrix()
        rot_x = utils.joint_rotation_x(metacarpal_basis, proximal_basis)
        rot_mat.set_rotation(metacarpal_basis.x_basis, rot_x)
        new_dir = rot_mat.transform_direction(metacarpal_basis.z_basis)

        rot_y = utils.joint_rotation_y(metacarpal_basis, proximal_basis)
        rot_mat.set_rotation(metacarpal_basis.y_basis, rot_y)
        new_dir = rot_mat.transform_direction(new_dir)

        # Add positions
        cur_pos += -new_dir * self.hand[num]["middle_finger"]["proximal"]["length"]

        # Construct further - currently at proximal bone top
        rot_x = utils.joint_rotation_x(proximal_basis, intermediate_basis)
        rot_mat.set_rotation(proximal_basis.x_basis, rot_x)
        new_dir = rot_mat.transform_direction(proximal_basis.z_basis)

        # Add positions
        cur_pos += -new_dir * self.hand[num]["middle_finger"]["intermediate"]["length"]

        # Compare
        diff = cur_pos.distance_to(real_pos)
        self.right_hand_middle_finger_intermediate_position_accuracy.append(diff)

        self.assertAlmostEqual(diff, 0., places=accuracy)

    @parameterized.expand([
        ["right_hand_middle_finger_intermediate_dh_position_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger positions, intermediate bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone top is compared to the real position and the difference is
        compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.
        """

        # Prepare bases
        basis = self.hand[num]["middle_finger"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["proximal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        proximal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare positions
        vec = self.hand[num]["middle_finger"]["intermediate"]["next_joint"]
        real_pos = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["proximal"]["next_joint"]
        proximal_next_joint = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["metacarpal"]["next_joint"]
        cur_pos = Leap.Vector(vec[0], vec[1], vec[2])

        # Transformations
        metacarpal_transform = Leap.Matrix(metacarpal_basis.x_basis,
                                           metacarpal_basis.y_basis,
                                           metacarpal_basis.z_basis,
                                           cur_pos)
        metacarpal_transform = metacarpal_transform.rigid_inverse()

        proximal_transform = Leap.Matrix(proximal_basis.x_basis,
                                         proximal_basis.y_basis,
                                         proximal_basis.z_basis,
                                         proximal_next_joint)
        proximal_transform = proximal_transform.rigid_inverse()

        # End position - "endeffector"
        end_pos_metacarpal = proximal_next_joint

        # Transform end position in current system
        vec = metacarpal_transform.transform_point(end_pos_metacarpal)
        l2 = self.hand[num]["middle_finger"]["proximal"]["length"]

        # From inverse kinematics (DH-convention)
        theta1 = -np.arctan(vec.x / vec.z)
        theta2 = -np.arcsin(vec.y / l2)

        # Construct the finger from the metacarpal bone top (next_joint)
        rot_theta1 = Leap.Matrix()
        rot_theta1.set_rotation(metacarpal_basis.y_basis, theta1)
        rot_theta2 = Leap.Matrix()
        rot_theta2.set_rotation(metacarpal_basis.x_basis, theta2)
        rot_m = rot_theta1 * rot_theta2
        new_dir = rot_m.transform_direction(metacarpal_basis.z_basis)

        # Add positions
        cur_pos += -new_dir * l2

        # End position - "endeffector"
        end_pos_proximal = real_pos

        # Transform end position in current system
        vec = proximal_transform.transform_point(end_pos_proximal)
        l3 = self.hand[num]["middle_finger"]["intermediate"]["length"]

        # From inverse kinematics (DH-convention)
        theta3 = -np.arcsin(vec.y / l3)

        # Construct further - currently at proximal bone top
        rot_theta3 = Leap.Matrix()
        rot_theta3.set_rotation(proximal_basis.x_basis, theta3)
        new_dir = rot_theta3.transform_direction(new_dir)

        # Add positions
        cur_pos += -new_dir * l3

        # Compare
        diff = cur_pos.distance_to(real_pos)
        self.right_hand_middle_finger_intermediate_dh_position_accuracy.append(diff)

        self.assertAlmostEqual(diff, 0., places=accuracy)

    @parameterized.expand([
        ["right_hand_middle_finger_distal_position_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger positions, distal bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone top is compared to the real position and the difference is
        compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.
        """

        # Prepare bases
        basis = self.hand[num]["middle_finger"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["proximal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        proximal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["intermediate"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        intermediate_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["distal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        distal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare positions
        vec = self.hand[num]["middle_finger"]["distal"]["next_joint"]
        real_pos = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["metacarpal"]["next_joint"]
        cur_pos = Leap.Vector(vec[0], vec[1], vec[2])

        # Construct the finger from the metacarpal bone top (next_joint)
        rot_mat = Leap.Matrix()
        rot_x = utils.joint_rotation_x(metacarpal_basis, proximal_basis)
        rot_mat.set_rotation(metacarpal_basis.x_basis, rot_x)
        new_dir = rot_mat.transform_direction(metacarpal_basis.z_basis)

        rot_y = utils.joint_rotation_y(metacarpal_basis, proximal_basis)
        rot_mat.set_rotation(metacarpal_basis.y_basis, rot_y)
        new_dir = rot_mat.transform_direction(new_dir)

        # Add positions
        cur_pos += -new_dir * self.hand[num]["middle_finger"]["proximal"]["length"]

        # Construct further - currently at proximal bone top
        rot_x = utils.joint_rotation_x(proximal_basis, intermediate_basis)
        rot_mat.set_rotation(proximal_basis.x_basis, rot_x)
        new_dir = rot_mat.transform_direction(proximal_basis.z_basis)

        # Add positions
        cur_pos += -new_dir * self.hand[num]["middle_finger"]["intermediate"]["length"]

        # Construct further - currently at intermediate bone top
        rot_x = utils.joint_rotation_x(intermediate_basis, distal_basis)
        rot_mat.set_rotation(intermediate_basis.x_basis, rot_x)
        new_dir = rot_mat.transform_direction(intermediate_basis.z_basis)

        # Add positions
        cur_pos += -new_dir * self.hand[num]["middle_finger"]["distal"]["length"]

        # Compare
        diff = cur_pos.distance_to(real_pos)
        self.right_hand_middle_finger_distal_position_accuracy.append(diff)

        self.assertAlmostEqual(diff, 0., places=accuracy)

    @parameterized.expand([
        ["right_hand_middle_finger_distal_dh_position_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger positions, distal bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone top is compared to the real position and the difference is
        compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.
        """

        # Prepare bases
        basis = self.hand[num]["middle_finger"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["proximal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        proximal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["middle_finger"]["intermediate"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        intermediate_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare positions
        vec = self.hand[num]["middle_finger"]["distal"]["next_joint"]
        real_pos = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["intermediate"]["next_joint"]
        intermediate_next_joint = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["proximal"]["next_joint"]
        proximal_next_joint = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["middle_finger"]["metacarpal"]["next_joint"]
        cur_pos = Leap.Vector(vec[0], vec[1], vec[2])

        # Transformations
        metacarpal_transform = Leap.Matrix(metacarpal_basis.x_basis,
                                           metacarpal_basis.y_basis,
                                           metacarpal_basis.z_basis,
                                           cur_pos)
        metacarpal_transform = metacarpal_transform.rigid_inverse()

        proximal_transform = Leap.Matrix(proximal_basis.x_basis,
                                         proximal_basis.y_basis,
                                         proximal_basis.z_basis,
                                         proximal_next_joint)
        proximal_transform = proximal_transform.rigid_inverse()

        intermediate_transform = Leap.Matrix(intermediate_basis.x_basis,
                                             intermediate_basis.y_basis,
                                             intermediate_basis.z_basis,
                                             intermediate_next_joint)
        intermediate_transform = intermediate_transform.rigid_inverse()

        # End position - "endeffector"
        end_pos_metacarpal = proximal_next_joint

        # Transform end position in current system
        vec = metacarpal_transform.transform_point(end_pos_metacarpal)
        l2 = self.hand[num]["middle_finger"]["proximal"]["length"]

        # From inverse kinematics (DH-convention)
        theta1 = -np.arctan(vec.x / vec.z)
        theta2 = -np.arcsin(vec.y / l2)

        # Construct the finger from the metacarpal bone top (next_joint)
        rot_theta1 = Leap.Matrix()
        rot_theta1.set_rotation(metacarpal_basis.y_basis, theta1)
        rot_theta2 = Leap.Matrix()
        rot_theta2.set_rotation(metacarpal_basis.x_basis, theta2)
        rot_m = rot_theta1 * rot_theta2
        new_dir = rot_m.transform_direction(metacarpal_basis.z_basis)

        # Add positions
        cur_pos += -new_dir * l2

        # End position - "endeffector"
        end_pos_proximal = intermediate_next_joint

        # Transform end position in current system
        vec = proximal_transform.transform_point(end_pos_proximal)
        l3 = self.hand[num]["middle_finger"]["intermediate"]["length"]

        # From inverse kinematics (DH-convention)
        theta3 = -np.arcsin(vec.y / l3)

        # Construct further - currently at proximal bone top
        rot_theta3 = Leap.Matrix()
        rot_theta3.set_rotation(proximal_basis.x_basis, theta3)
        new_dir = rot_theta3.transform_direction(new_dir)

        # Add positions
        cur_pos += -new_dir * l3

        # End position - "endeffector"
        end_pos_intermediate = real_pos

        # Transform end position in current system
        vec = intermediate_transform.transform_point(end_pos_intermediate)
        l4 = self.hand[num]["middle_finger"]["distal"]["length"]

        # From inverse kinematics (DH-convention)
        theta4 = -np.arcsin(vec.y / l4)

        # Construct further - currently at intermediate bone top
        rot_theta4 = Leap.Matrix()
        rot_theta4.set_rotation(proximal_basis.x_basis, theta4)
        new_dir = rot_theta4.transform_direction(new_dir)

        # Add positions
        cur_pos += -new_dir * l4

        # Compare
        diff = cur_pos.distance_to(real_pos)
        self.right_hand_middle_finger_distal_dh_position_accuracy.append(diff)

        self.assertAlmostEqual(diff, 0., places=accuracy)
