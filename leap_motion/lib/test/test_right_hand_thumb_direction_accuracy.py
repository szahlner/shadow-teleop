import os
import unittest
from parameterized import parameterized
import matplotlib.pyplot as plt
import numpy as np

from lib.leap_motion import Leap
from lib.test import test_utils
from lib.test.argument_parser import wrapper
from lib import utils


class TestRightHandThumbDirectionAccuracy(unittest.TestCase):
    right_hand_thumb_proximal_dh_direction_x_accuracy = []
    right_hand_thumb_proximal_dh_direction_y_accuracy = []
    right_hand_thumb_proximal_dh_direction_z_accuracy = []
    right_hand_thumb_intermediate_dh_direction_x_accuracy = []
    right_hand_thumb_intermediate_dh_direction_y_accuracy = []
    right_hand_thumb_intermediate_dh_direction_z_accuracy = []
    right_hand_thumb_distal_dh_direction_x_accuracy = []
    right_hand_thumb_distal_dh_direction_y_accuracy = []
    right_hand_thumb_distal_dh_direction_z_accuracy = []

    @classmethod
    def tearDownClass(cls):
        if test_utils.TEST_NUMBER_DATA_SAMPLES == 0:
            return

        with plt.style.context('ggplot'):
            face_color = (0.8500, 0.3250, 0.0980, 1.)

            # Plot 1
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9), constrained_layout=True)

            axes[0][0].hist(cls.right_hand_thumb_proximal_dh_direction_x_accuracy, bins=50, facecolor=face_color)
            axes[0][0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[0][0].set_ylabel("Quantity")
            axes[0][0].set_title("Proximal x", fontsize=12)

            axes[0][1].hist(cls.right_hand_thumb_proximal_dh_direction_y_accuracy, bins=50, facecolor=face_color)
            axes[0][1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[0][1].set_xlabel("Distance error in mm", labelpad=20)
            axes[0][1].set_title("Proximal y", fontsize=12)

            axes[0][2].hist(cls.right_hand_thumb_proximal_dh_direction_z_accuracy, bins=50, facecolor=face_color)
            axes[0][2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[0][2].set_title("Proximal z", fontsize=12)

            axes[1][0].hist(cls.right_hand_thumb_intermediate_dh_direction_x_accuracy, bins=50, facecolor=face_color)
            axes[1][0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[1][0].set_ylabel("Quantity")
            axes[1][0].set_title("Intermediate x", fontsize=12)

            axes[1][1].hist(cls.right_hand_thumb_intermediate_dh_direction_y_accuracy, bins=50, facecolor=face_color)
            axes[1][1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[1][1].set_xlabel("Distance error in mm", labelpad=20)
            axes[1][1].set_title("Intermediate y", fontsize=12)

            axes[1][2].hist(cls.right_hand_thumb_intermediate_dh_direction_z_accuracy, bins=50, facecolor=face_color)
            axes[1][2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[1][2].set_title("Intermediate z", fontsize=12)

            axes[2][0].hist(cls.right_hand_thumb_distal_dh_direction_x_accuracy, bins=50, facecolor=face_color)
            axes[2][0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[2][0].set_ylabel("Quantity")
            axes[2][0].set_title("Distal x", fontsize=12)

            axes[2][1].hist(cls.right_hand_thumb_distal_dh_direction_y_accuracy, bins=50, facecolor=face_color)
            axes[2][1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[2][1].set_xlabel("Distance error in mm", labelpad=20)
            axes[2][1].set_title("Distal y", fontsize=12)

            axes[2][2].hist(cls.right_hand_thumb_distal_dh_direction_z_accuracy, bins=50, facecolor=face_color)
            axes[2][2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[2][2].set_title("Distal z", fontsize=12)

            fig.suptitle("Right Hand Thumb DH Bone Direction, samples: {}".format(len(cls.right_hand_thumb_proximal_dh_direction_x_accuracy)), fontsize=14)

            fig.savefig(os.path.join(test_utils.TEST_DATA_RESULTS_DIRECTORY, "right_hand_thumb_dh_direction_accuracy.png"), format="png")
            plt.close()

            test_utils.right_hand_thumb_proximal_dh_direction_x_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_proximal_dh_direction_x_accuracy))
            test_utils.right_hand_thumb_proximal_dh_direction_y_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_proximal_dh_direction_y_accuracy))
            test_utils.right_hand_thumb_proximal_dh_direction_z_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_proximal_dh_direction_z_accuracy))

            test_utils.right_hand_thumb_intermediate_dh_direction_x_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_intermediate_dh_direction_x_accuracy))
            test_utils.right_hand_thumb_intermediate_dh_direction_y_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_intermediate_dh_direction_y_accuracy))
            test_utils.right_hand_thumb_intermediate_dh_direction_z_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_intermediate_dh_direction_z_accuracy))

            test_utils.right_hand_thumb_distal_dh_direction_x_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_distal_dh_direction_x_accuracy))
            test_utils.right_hand_thumb_distal_dh_direction_y_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_distal_dh_direction_y_accuracy))
            test_utils.right_hand_thumb_distal_dh_direction_z_accuracy_mean = np.abs(np.mean(cls.right_hand_thumb_distal_dh_direction_z_accuracy))

    @classmethod
    def setUpClass(cls):
        if test_utils.TEST_NUMBER_DATA_SAMPLES != 0:
            cls.hand = test_utils.load_hand()

    def setUp(self):
        if test_utils.TEST_NUMBER_DATA_SAMPLES == 0:
            self.skipTest("no sample data available")

    @parameterized.expand([
        ["right_hand_thumb_proximal_dh_direction_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger direction, proximal bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone direction is compared to the real direction and the difference
        is compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.

        Note
        ----
            The compared direction is inverted, this compares the z-direction of the bone.
        """

        # Prepare bases
        basis = self.hand[num]["hand"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        hand_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["thumb"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare directions
        basis = self.hand[num]["thumb"]["proximal"]["basis"]
        real_dir = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])

        # Prepare positions
        vec = self.hand[num]["thumb"]["metacarpal"]["next_joint"]
        start_pos_metacarpal = Leap.Vector(vec[0], vec[1], vec[2])

        # Compute new metacarpal basis
        rot_angle = utils.joint_rotation_y(hand_basis, metacarpal_basis) - 45 * Leap.DEG_TO_RAD
        rot_m = Leap.Matrix()
        rot_m.set_rotation(hand_basis.y_basis, rot_angle)
        x_basis = rot_m.transform_direction(hand_basis.x_basis)
        y_basis = hand_basis.y_basis
        z_basis = rot_m.transform_direction(hand_basis.z_basis)

        metacarpal_basis = Leap.Matrix(x_basis,
                                       y_basis,
                                       z_basis,
                                       start_pos_metacarpal)
        metacarpal_transform = metacarpal_basis.rigid_inverse()

        # End position - "endeffector"
        vec = self.hand[num]["thumb"]["proximal"]["next_joint"]
        end_pos_metacarpal = Leap.Vector(vec[0], vec[1], vec[2])

        # Transform end position in current system
        vec = metacarpal_transform.transform_point(end_pos_metacarpal)
        l2 = self.hand[num]["thumb"]["proximal"]["length"]

        # From inverse kinematics (DH-convention)
        theta1 = -np.arctan2(vec.x, vec.y) + np.pi
        theta2 = np.pi - np.arccos(vec.z / l2)

        # Construct the finger from the metacarpal bone top (next_joint)
        rot_m.set_rotation(z_basis, -theta1)
        x_basis = rot_m.transform_direction(x_basis)
        z_basis = rot_m.transform_direction(z_basis)

        rot_m.set_rotation(x_basis, theta2)
        cur_dir = rot_m.transform_direction(z_basis)

        # Compare
        diff_x = real_dir.x - cur_dir.x
        diff_y = real_dir.y - cur_dir.y
        diff_z = real_dir.z - cur_dir.z

        self.right_hand_thumb_proximal_dh_direction_x_accuracy.append(diff_x)
        self.right_hand_thumb_proximal_dh_direction_y_accuracy.append(diff_y)
        self.right_hand_thumb_proximal_dh_direction_z_accuracy.append(diff_z)

        self.assertAlmostEqual(diff_x, 0., places=accuracy, msg="x-component")
        self.assertAlmostEqual(diff_y, 0., places=accuracy, msg="y-component")
        self.assertAlmostEqual(diff_z, 0., places=accuracy, msg="z-component")

    @parameterized.expand([
        ["right_hand_thumb_intermediate_dh_direction_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger direction, intermediate bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone direction is compared to the real direction and the difference
        is compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.

        Note
        ----
            The compared direction is inverted, this compares the z-direction of the bone.
        """

        # Prepare bases
        basis = self.hand[num]["hand"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        hand_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["thumb"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare directions
        basis = self.hand[num]["thumb"]["intermediate"]["basis"]
        real_dir = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])

        # Prepare positions
        vec = self.hand[num]["thumb"]["metacarpal"]["next_joint"]
        start_pos_metacarpal = Leap.Vector(vec[0], vec[1], vec[2])

        # Compute new metacarpal basis
        rot_angle = utils.joint_rotation_y(hand_basis, metacarpal_basis) - 45 * Leap.DEG_TO_RAD
        rot_m = Leap.Matrix()
        rot_m.set_rotation(hand_basis.y_basis, rot_angle)
        x_basis = rot_m.transform_direction(hand_basis.x_basis)
        y_basis = hand_basis.y_basis
        z_basis = rot_m.transform_direction(hand_basis.z_basis)

        metacarpal_basis = Leap.Matrix(x_basis,
                                       y_basis,
                                       z_basis,
                                       start_pos_metacarpal)
        metacarpal_transform = metacarpal_basis.rigid_inverse()

        # End position - "endeffector"
        vec = self.hand[num]["thumb"]["proximal"]["next_joint"]
        end_pos_metacarpal = Leap.Vector(vec[0], vec[1], vec[2])

        # Transform end position in current system
        vec = metacarpal_transform.transform_point(end_pos_metacarpal)
        l2 = self.hand[num]["thumb"]["proximal"]["length"]

        # From inverse kinematics (DH-convention)
        theta1 = -np.arctan2(vec.x, vec.y) + np.pi
        theta2 = np.pi - np.arccos(vec.z / l2)

        # Compute new proximal basis
        rot_m.set_rotation(z_basis, -theta1)
        x_basis = rot_m.transform_direction(x_basis)
        y_basis = rot_m.transform_direction(y_basis)

        rot_m.set_rotation(x_basis, theta2)
        y_basis = rot_m.transform_direction(y_basis)
        z_basis = rot_m.transform_direction(z_basis)

        proximal_basis = Leap.Matrix(x_basis,
                                     y_basis,
                                     z_basis,
                                     end_pos_metacarpal)
        proximal_transform = proximal_basis.rigid_inverse()

        # End position - "endeffector"
        vec = self.hand[num]["thumb"]["intermediate"]["next_joint"]
        end_pos_proximal = Leap.Vector(vec[0], vec[1], vec[2])

        # Transform end position in current system
        vec = proximal_transform.transform_point(end_pos_proximal)
        l4 = self.hand[num]["thumb"]["intermediate"]["length"]

        # From inverse kinematics (DH-convention)
        theta3 = np.arctan(vec.y / vec.z)
        theta4 = np.arcsin(vec.x / l4)

        # Construct the finger from the metacarpal bone top (next_joint)
        # Compute new intermediate basis
        rot_m.set_rotation(x_basis, theta3)
        y_basis = rot_m.transform_direction(y_basis)
        z_basis = rot_m.transform_direction(z_basis)

        rot_m.set_rotation(y_basis, theta4)
        cur_dir = rot_m.transform_direction(z_basis)

        # Compare
        diff_x = real_dir.x - cur_dir.x
        diff_y = real_dir.y - cur_dir.y
        diff_z = real_dir.z - cur_dir.z

        self.right_hand_thumb_intermediate_dh_direction_x_accuracy.append(diff_x)
        self.right_hand_thumb_intermediate_dh_direction_y_accuracy.append(diff_y)
        self.right_hand_thumb_intermediate_dh_direction_z_accuracy.append(diff_z)

        self.assertAlmostEqual(diff_x, 0., places=accuracy, msg="x-component")
        self.assertAlmostEqual(diff_y, 0., places=accuracy, msg="y-component")
        self.assertAlmostEqual(diff_z, 0., places=accuracy, msg="z-component")

    @parameterized.expand([
        ["right_hand_thumb_distal_dh_direction_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks how accurate the calculated joint angles match the real finger direction, distal bone.

        This starts to build up a finger from the top of the metacarpal bone corresponding to the bone lengths,
        directions and calculated joint angles. Each bone direction is compared to the real direction and the difference
        is compared to zero (rounded by the given number of decimal places).

        Parameters
        ----------
        name : string
            Name of the function.
        num : int
            Sample number.
        accuracy : int
            Specifies the decimal places to be rounded to.

        Note
        ----
            The compared direction is inverted, this compares the z-direction of the bone.
        """

        # Prepare bases
        basis = self.hand[num]["hand"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        hand_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        basis = self.hand[num]["thumb"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare directions
        basis = self.hand[num]["thumb"]["distal"]["basis"]
        real_dir = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])

        # Prepare positions
        vec = self.hand[num]["thumb"]["metacarpal"]["next_joint"]
        start_pos_metacarpal = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["hand"]["palm_position"]
        palm_position = Leap.Vector(vec[0], vec[1], vec[2])

        # Compute new metacarpal basis
        rot_angle = utils.joint_rotation_y(hand_basis, metacarpal_basis) - 45 * Leap.DEG_TO_RAD
        rot_m = Leap.Matrix()
        rot_m.set_rotation(hand_basis.y_basis, rot_angle)
        x_basis = rot_m.transform_direction(hand_basis.x_basis)
        y_basis = hand_basis.y_basis
        z_basis = rot_m.transform_direction(hand_basis.z_basis)

        metacarpal_basis = Leap.Matrix(x_basis,
                                       y_basis,
                                       z_basis,
                                       start_pos_metacarpal)
        metacarpal_transform = metacarpal_basis.rigid_inverse()

        # End position - "endeffector"
        vec = self.hand[num]["thumb"]["proximal"]["next_joint"]
        end_pos_metacarpal = Leap.Vector(vec[0], vec[1], vec[2])

        # Transform end position in current system
        vec = metacarpal_transform.transform_point(end_pos_metacarpal)
        l2 = self.hand[num]["thumb"]["proximal"]["length"]

        # From inverse kinematics (DH-convention)
        theta1 = -np.arctan2(vec.x, vec.y) + np.pi
        theta2 = np.pi - np.arccos(vec.z / l2)

        # Compute new proximal basis
        rot_m.set_rotation(z_basis, -theta1)
        x_basis = rot_m.transform_direction(x_basis)
        y_basis = rot_m.transform_direction(y_basis)

        rot_m.set_rotation(x_basis, theta2)
        y_basis = rot_m.transform_direction(y_basis)
        z_basis = rot_m.transform_direction(z_basis)

        proximal_basis = Leap.Matrix(x_basis,
                                     y_basis,
                                     z_basis,
                                     end_pos_metacarpal)
        proximal_transform = proximal_basis.rigid_inverse()

        # End position - "endeffector"
        vec = self.hand[num]["thumb"]["intermediate"]["next_joint"]
        end_pos_proximal = Leap.Vector(vec[0], vec[1], vec[2])

        # Transform end position in current system
        vec = proximal_transform.transform_point(end_pos_proximal)
        l4 = self.hand[num]["thumb"]["intermediate"]["length"]

        # From inverse kinematics (DH-convention)
        theta3 = np.arctan(vec.y / vec.z)
        theta4 = np.arcsin(vec.x / l4)

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
                                         end_pos_proximal)
        intermediate_transform = intermediate_basis.rigid_inverse()

        # End position - "endeffector"
        vec = self.hand[num]["thumb"]["distal"]["next_joint"]
        end_pos_intermediate = Leap.Vector(vec[0], vec[1], vec[2])

        # Transform end position in current system
        vec = intermediate_transform.transform_point(end_pos_intermediate)
        l5 = self.hand[num]["thumb"]["distal"]["length"]

        # From inverse kinematics (DH-convention)
        theta5 = np.pi - np.arccos(vec.z / l5)

        # Construct the finger from the metacarpal bone top (next_joint)
        rot_m.set_rotation(y_basis, theta5)
        x_basis = rot_m.transform_direction(x_basis)
        cur_dir = rot_m.transform_direction(z_basis)

        # Compare
        diff_x = real_dir.x - cur_dir.x
        diff_y = real_dir.y - cur_dir.y
        diff_z = real_dir.z - cur_dir.z

        self.right_hand_thumb_distal_dh_direction_x_accuracy.append(diff_x)
        self.right_hand_thumb_distal_dh_direction_y_accuracy.append(diff_y)
        self.right_hand_thumb_distal_dh_direction_z_accuracy.append(diff_z)

        self.assertAlmostEqual(diff_x, 0., places=accuracy, msg="x-component")
        self.assertAlmostEqual(diff_y, 0., places=accuracy, msg="y-component")
        self.assertAlmostEqual(diff_z, 0., places=accuracy, msg="z-component")
