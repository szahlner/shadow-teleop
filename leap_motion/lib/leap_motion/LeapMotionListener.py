import os
import pybullet
import numpy as np
import imageio

from . import Leap
from lib.test import test_utils
from lib import utils


from time import time
milliseconds = lambda: int(time() * 1000)


class LeapMotionListener(Leap.Listener):

    def __init__(self, num_samples=None, with_object=False, render=False):
        super().__init__()

        self.t_ = milliseconds()

        # Sample / Test mode
        self.num_samples = num_samples
        self.samples = 0

        # Device
        self.device = None

        # Clear sample data
        if num_samples is not None:
            test_utils.clear_samples()

        # Render
        self.render = render
        if render:
            self.images = []
            # Camera defaults
            self.camera_view_matrix = pybullet.computeViewMatrix(cameraEyePosition=[0.25, -0.75, 0.5],
                                                                 cameraTargetPosition=[0., 0., 0.2],
                                                                 cameraUpVector=[0., 0., 1.])
            self.camera_projection_matrix = pybullet.computeProjectionMatrixFOV(fov=45., aspect=1., nearVal=0.1,
                                                                                farVal=1.1)

        # PYBULLET SETUP
        # Connect to physics client, use graphical interface (GUI)
        self.physics_client = pybullet.connect(pybullet.GUI)

        # Load (right) shadow hand and place it at origin
        self.shadow_hand_right_urdf = os.path.join("assets", "urdf", "shadow_hand_right.urdf")
        self.shadow_hand_right_id = pybullet.loadURDF(fileName=self.shadow_hand_right_urdf,
                                                      basePosition=[0., 0., 0.],
                                                      baseOrientation=pybullet.getQuaternionFromEuler([0., 0., 0.]),
                                                      flags=pybullet.URDF_USE_SELF_COLLISION)

        # Object
        self.with_object = with_object
        if with_object:
            # Load cube
            pybullet.setGravity(0., 9.81, 0.)

            object_path = os.path.join("assets", "obj", "block.obj")
            visual_shape_id = pybullet.createVisualShape(fileName=object_path,
                                                         shapeType=pybullet.GEOM_MESH,
                                                         rgbaColor=None,
                                                         meshScale=[0.02, 0.02, 0.02])

            collision_shape_id = pybullet.createCollisionShape(fileName=object_path,
                                                               shapeType=pybullet.GEOM_MESH,
                                                               meshScale=[0.02, 0.02, 0.02])

            object_texture_path = os.path.join("assets", "materials", "textures", "block.png")
            texture_id = pybullet.loadTexture(object_texture_path)

            object_id = pybullet.createMultiBody(baseMass=0.1,
                                                 baseVisualShapeIndex=visual_shape_id,
                                                 baseCollisionShapeIndex=collision_shape_id,
                                                 basePosition=[0., -0.07, 0.35],
                                                 baseOrientation=pybullet.getQuaternionFromEuler([0., 0., 0.]))

            pybullet.changeVisualShape(object_id, -1, textureUniqueId=texture_id)

        # Number of joints in (right) shadow hand and target position array
        self.shadow_hand_right_num_joints = pybullet.getNumJoints(self.shadow_hand_right_id)
        self.shadow_hand_right_target_pos = [0.] * self.shadow_hand_right_num_joints

        # Set sub-step-parameter and start Real-Time-Simulation
        pybullet.setPhysicsEngineParameter(numSubSteps=0)
        pybullet.setRealTimeSimulation(1)

    def __del__(self):

        # Disconnect from physics client
        try:
            pybullet.disconnect()
        except pybullet.error:
            pass

        super().__del__()

    def on_init(self, controller):
        print("Listener Initialized")

    def on_connect(self, controller):
        self.device = controller.devices[0]
        print("Connected to device: {}".format(self.device.serial_number))

    def on_disconnect(self, controller):
        print("Disconnected from device: {}".format(self.device.serial_number))
        self.device = None

    def on_exit(self, controller):
        print("Listener Exited")

        if self.render:
            gif_path = os.path.join("images", "shadow_hand.gif")
            imageio.mimsave(gif_path, [np.array(img) for n, img in enumerate(self.images) if n % 3 == 0], fps=10)

    def on_frame(self, controller):

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
                    theta1, theta2, theta3, theta4, theta5 = utils.thumb_joint_rotations(hand)

                    if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3) and not np.isnan(
                            theta4) and not np.isnan(theta5):
                        self.shadow_hand_right_target_pos[24:29] = [
                            np.clip(theta1, -1.047, 1.047),
                            np.clip(theta2, 0., 1.222),
                            np.clip(theta3, -0.209, 0.209),
                            np.clip(theta4, -0.524, 0.524),
                            np.clip(theta5, 0., 1.571)]

                    # INDEX finger
                    index_finger = hand.fingers.finger_type(Leap.Finger.TYPE_INDEX)[0]

                    theta1, theta2, theta3, theta4 = utils.finger_joint_rotations(index_finger)

                    if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3) and not np.isnan(theta4):
                        self.shadow_hand_right_target_pos[3:7] = [
                            np.clip(theta1, -0.349, 0.349),
                            np.clip(theta2, 0., 1.571),
                            np.clip(theta3, 0., 1.571),
                            np.clip(theta4, 0., 1.571)]

                    """
                    # Get the Finger Object and the bases of each bone in this finger
                    metacarpal_basis = index_finger.bone(Leap.Bone.TYPE_METACARPAL).basis
                    proximal_basis = index_finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
                    intermediate_basis = index_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
                    distal_basis = index_finger.bone(Leap.Bone.TYPE_DISTAL).basis

                    # Compute the rotation angles of the INDEX finger
                    self.shadow_hand_right_target_pos[3:7] = [
                        utils.joint_rotation_y(metacarpal_basis, proximal_basis, -0.349, 0.349),
                        utils.joint_rotation_x(metacarpal_basis, proximal_basis, 0., 1.571),
                        utils.joint_rotation_x(proximal_basis, intermediate_basis, 0., 1.571),
                        utils.joint_rotation_x(intermediate_basis, distal_basis, 0., 1.571)]
                    
                    # Update the rotation angles of the INDEX finger
                    pybullet.setJointMotorControlArray(bodyUniqueId=self.shadow_hand_right_id,
                                                       jointIndices=range(self.shadow_hand_right_num_joints),
                                                       controlMode=pybullet.POSITION_CONTROL,
                                                       targetPositions=self.shadow_hand_right_target_pos)
                    """

                    # MIDDLE finger
                    middle_finger = hand.fingers.finger_type(Leap.Finger.TYPE_MIDDLE)[0]

                    theta1, theta2, theta3, theta4 = utils.finger_joint_rotations(middle_finger)

                    if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3) and not np.isnan(theta4):
                        self.shadow_hand_right_target_pos[8:12] = [
                            np.clip(theta1, -0.349, 0.349),
                            np.clip(theta2, 0., 1.571),
                            np.clip(theta3, 0., 1.571),
                            np.clip(theta4, 0., 1.571)]

                    """
                    # Get the Finger Object and the bases of each bone in this finger
                    middle_finger = hand.fingers.finger_type(Leap.Finger.TYPE_MIDDLE)[0]
                    metacarpal_basis = middle_finger.bone(Leap.Bone.TYPE_METACARPAL).basis
                    proximal_basis = middle_finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
                    intermediate_basis = middle_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
                    distal_basis = middle_finger.bone(Leap.Bone.TYPE_DISTAL).basis

                    # Compute the rotation angles of the MIDDLE finger
                    self.shadow_hand_right_target_pos[8:12] = [
                        utils.joint_rotation_y(metacarpal_basis, proximal_basis, -0.349, 0.349),
                        utils.joint_rotation_x(metacarpal_basis, proximal_basis, 0., 1.571),
                        utils.joint_rotation_x(proximal_basis, intermediate_basis, 0., 1.571),
                        utils.joint_rotation_x(intermediate_basis, distal_basis, 0., 1.571)]

                    # Update the rotation angles of the MIDDLE finger
                    pybullet.setJointMotorControlArray(bodyUniqueId=self.shadow_hand_right_id,
                                                       jointIndices=range(self.shadow_hand_right_num_joints),
                                                       controlMode=pybullet.POSITION_CONTROL,
                                                       targetPositions=self.shadow_hand_right_target_pos)
                    """
                    # RING finger
                    ring_finger = hand.fingers.finger_type(Leap.Finger.TYPE_RING)[0]

                    theta1, theta2, theta3, theta4 = utils.finger_joint_rotations(ring_finger)

                    if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3) and not np.isnan(theta4):
                        self.shadow_hand_right_target_pos[13:17] = [
                            -np.clip(theta1, -0.349, 0.349),  # "-" = BUG ?
                            np.clip(theta2, 0., 1.571),
                            np.clip(theta3, 0., 1.571),
                            np.clip(theta4, 0., 1.571)]

                    """
                    # Get the Finger Object and the bases of each bone in this finger
                    ring_finger = hand.fingers.finger_type(Leap.Finger.TYPE_RING)[0]
                    metacarpal_basis = ring_finger.bone(Leap.Bone.TYPE_METACARPAL).basis
                    proximal_basis = ring_finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
                    intermediate_basis = ring_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
                    distal_basis = ring_finger.bone(Leap.Bone.TYPE_DISTAL).basis

                    # Compute the rotation angles of the RING finger
                    self.shadow_hand_right_target_pos[13:17] = [
                        -utils.joint_rotation_y(metacarpal_basis, proximal_basis, -0.349, 0.349),  # "-" = BUG ?
                        utils.joint_rotation_x(metacarpal_basis, proximal_basis, 0., 1.571),
                        utils.joint_rotation_x(proximal_basis, intermediate_basis, 0., 1.571),
                        utils.joint_rotation_x(intermediate_basis, distal_basis, 0., 1.571)]

                    # Update the rotation angles of the RING finger
                    pybullet.setJointMotorControlArray(bodyUniqueId=self.shadow_hand_right_id,
                                                       jointIndices=range(self.shadow_hand_right_num_joints),
                                                       controlMode=pybullet.POSITION_CONTROL,
                                                       targetPositions=self.shadow_hand_right_target_pos)
                    """
                    # LITTLE finger
                    little_finger = hand.fingers.finger_type(Leap.Finger.TYPE_PINKY)[0]

                    theta1, theta2, theta3, theta4 = utils.finger_joint_rotations(little_finger)
                    # theta5 = utils.little_finger_5_joint_rotation(little_finger) Little Finger 5 = 0

                    if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3) and not np.isnan(theta4):
                        self.shadow_hand_right_target_pos[19:23] = [
                            # Little Finger 5 = 0 np.clip(theta5, 0., 0.785),
                            -np.clip(theta1, -0.349, 0.349),  # "-" = BUG ?
                            np.clip(theta2, 0., 1.571),
                            np.clip(theta3, 0., 1.571),
                            np.clip(theta4, 0., 1.571)]

                    """
                    # Get the Finger Object and the bases of each bone in this finger
                    metacarpal_basis = little_finger.bone(Leap.Bone.TYPE_METACARPAL).basis
                    proximal_basis = little_finger.bone(Leap.Bone.TYPE_PROXIMAL).basis
                    intermediate_basis = little_finger.bone(Leap.Bone.TYPE_INTERMEDIATE).basis
                    distal_basis = little_finger.bone(Leap.Bone.TYPE_DISTAL).basis

                    # Compute the rotation angles of the LITTLE finger
                    self.shadow_hand_right_target_pos[19:23] = [
                        # Little Finger 5
                        -utils.joint_rotation_y(metacarpal_basis, proximal_basis, -0.349, 0.349),  # "-" = BUG ?
                        utils.joint_rotation_x(metacarpal_basis, proximal_basis, 0., 1.571),
                        utils.joint_rotation_x(proximal_basis, intermediate_basis, 0., 1.571),
                        utils.joint_rotation_x(intermediate_basis, distal_basis, 0., 1.571)]

                    # Update the rotation angles of the LITTLE finger
                    pybullet.setJointMotorControlArray(bodyUniqueId=self.shadow_hand_right_id,
                                                       jointIndices=range(self.shadow_hand_right_num_joints),
                                                       controlMode=pybullet.POSITION_CONTROL,
                                                       targetPositions=self.shadow_hand_right_target_pos)
                    """

                    # WRIST
                    self.shadow_hand_right_target_pos[1:3] = [
                        -utils.joint_rotation_y(hand.arm.basis, hand.basis, -0.489, 0.140),
                        utils.joint_rotation_x(hand.arm.basis, hand.basis, -0.698, 0.489)]

                    # Update the rotation angles of the WRIST
                    pybullet.setJointMotorControlArray(bodyUniqueId=self.shadow_hand_right_id,
                                                       jointIndices=range(self.shadow_hand_right_num_joints),
                                                       controlMode=pybullet.POSITION_CONTROL,
                                                       targetPositions=self.shadow_hand_right_target_pos)

                    # In sample mode take the first num_samples frames as samples
                    # Continue normally otherwise / afterwards.
                    if self.num_samples is not None and self.samples < int(self.num_samples):
                        self.samples += 1
                        test_utils.save_hand(hand, self.num_samples, hand.is_right)
                        print("Samples taken: {}".format(self.samples))
                    elif self.num_samples is not None and self.samples == int(self.num_samples):
                        print("Ended sampling process. Took {} samples.".format(self.samples))
                        self.samples += 1

                    if self.render:
                        t_now = milliseconds()
                        #print(t_now)
                        if self.t_ + 3 <= t_now:
                            self.t_ = t_now

                            img = pybullet.getCameraImage(width=512, height=512,
                                                          viewMatrix=self.camera_view_matrix,
                                                          projectionMatrix=self.camera_projection_matrix)
                            img = img[2]
                            img = img[:, :, :-1]
                            img[np.where((img == [255, 255, 255]).all(axis=2))] = [101, 158, 199]
                            self.images.append(img)

                    # Mark right hand as processed
                    right_hand_processed = True
