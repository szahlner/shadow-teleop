import os
import unittest
from parameterized import parameterized
import matplotlib.pyplot as plt
import numpy as np

from lib.leap_motion import Leap
from lib.test import test_utils
from lib.test.argument_parser import wrapper


class TestRightHandLittleFinger5PositionAccuracy(unittest.TestCase):
    right_hand_little_finger_5_dh_position_accuracy = []
    right_hand_little_finger_5_dh_position_value = []

    @classmethod
    def tearDownClass(cls):
        if test_utils.TEST_NUMBER_DATA_SAMPLES == 0:
            return

        with plt.style.context('ggplot'):
            face_color = (0.4660, 0.6740, 0.1880, 1.)

            # Plot 1
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3), constrained_layout=True)

            axes[0].hist(cls.right_hand_little_finger_5_dh_position_accuracy, bins=50, facecolor=face_color)
            axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[0].set_xlabel("Distance error in mm", labelpad=20)
            axes[0].set_ylabel("Quantity")

            axes[1].hist(cls.right_hand_little_finger_5_dh_position_value, bins=50, facecolor=face_color)
            axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[1].set_xlabel("Angle in degree", labelpad=20)
            axes[1].set_ylabel("Quantity")

            fig.suptitle("Right Hand Little Finger 5 Bone End-Position, samples: {}".format(len(cls.right_hand_little_finger_5_dh_position_accuracy)), fontsize=14)

            fig.savefig(os.path.join(test_utils.TEST_DATA_RESULTS_DIRECTORY, "right_hand_little_finger_5_dh_position_accuracy.png"), format="png")
            plt.close()

    @classmethod
    def setUpClass(cls):
        if test_utils.TEST_NUMBER_DATA_SAMPLES != 0:
            cls.hand = test_utils.load_hand()

    def setUp(self):
        if test_utils.TEST_NUMBER_DATA_SAMPLES == 0:
            self.skipTest("no sample data available")

    @parameterized.expand([
        ["right_hand_little_finger_5_dh_position_accuracy_{}".format(x), x, wrapper.args["accuracy"]]
        for x in range(test_utils.TEST_NUMBER_DATA_SAMPLES)],
        skip_on_empty=True)
    def test_sequence(self, name, num, accuracy):
        """
        Checks little finger 5 accuracy and necessity.

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
        basis = self.hand[num]["little_finger"]["metacarpal"]["basis"]
        x_basis = Leap.Vector(basis[0, 0], basis[0, 1], basis[0, 2])
        y_basis = Leap.Vector(basis[1, 0], basis[1, 1], basis[1, 2])
        z_basis = Leap.Vector(basis[2, 0], basis[2, 1], basis[2, 2])
        metacarpal_basis = Leap.Matrix(x_basis, y_basis, z_basis, Leap.Vector(0., 0., 0.))

        # Prepare positions
        vec = self.hand[num]["little_finger"]["metacarpal"]["next_joint"]
        real_pos = Leap.Vector(vec[0], vec[1], vec[2])

        vec = self.hand[num]["little_finger"]["metacarpal"]["prev_joint"]
        cur_pos = Leap.Vector(vec[0], vec[1], vec[2])

        # Transformation
        metacarpal_transform = Leap.Matrix(metacarpal_basis.x_basis,
                                           metacarpal_basis.y_basis,
                                           metacarpal_basis.z_basis,
                                           cur_pos)
        metacarpal_transform = metacarpal_transform.rigid_inverse()

        vec = metacarpal_transform.transform_point(real_pos)

        l1 = self.hand[num]["little_finger"]["metacarpal"]["length"]

        theta = np.pi - np.arccos((np.cos(7 * np.pi / 36) * np.cos(7 * np.pi / 36) * l1 + vec.z) / (
                np.cos(11 * np.pi / 36) * np.sin(7 * np.pi / 36) * l1))

        if np.isnan(theta):
            self.skipTest("little finger 5 nan")

        self.right_hand_little_finger_5_dh_position_value.append(theta * Leap.RAD_TO_DEG)

        # Construct the finger from the metacarpal bone bottom (prev_joint)
        rot_m = Leap.Matrix()
        rot_m.set_rotation(metacarpal_basis.y_basis, 35 * Leap.DEG_TO_RAD)
        rot_axis_x = rot_m.transform_direction(metacarpal_basis.x_basis)
        rot_axis_z = rot_m.transform_direction(metacarpal_basis.z_basis)

        rot_m.set_rotation(rot_axis_x, np.pi)
        rot_axis_z = rot_m.transform_direction(rot_axis_z)

        rot_m.set_rotation(rot_axis_z, theta)
        new_dir = rot_m.transform_direction(metacarpal_basis.z_basis)

        # Add positions
        cur_pos += -new_dir * l1

        # Compare
        diff = cur_pos.distance_to(real_pos)
        self.right_hand_little_finger_5_dh_position_accuracy.append(diff)

        self.assertAlmostEqual(diff, 0., places=accuracy)