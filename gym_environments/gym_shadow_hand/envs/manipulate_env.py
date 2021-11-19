import os
import numpy as np
import pybullet
from gym import spaces
from gym import utils
from collections import OrderedDict
from gym_shadow_hand.envs.pybullet_env import PyBulletEnv, PyBulletGoalEnv
from gym_shadow_hand.envs import rotations
from scipy.spatial.transform import Rotation as R

MODEL_PATH = "shadow_hand_right.urdf"
BLOCK_PATH = "block.obj"
BLOCK_TEXTURE_PATH = "block.png"
BLOCK_LENGTH = 0.04 # Block length in metres

MOVABLE_JOINTS = [b"rh_FFJ4", b"rh_FFJ3", b"rh_FFJ2",
                  b"rh_MFJ4", b"rh_MFJ3", b"rh_MFJ2",
                  b"rh_RFJ4", b"rh_RFJ3", b"rh_RFJ2",
                  b"rh_LFJ5", b"rh_LFJ4", b"rh_LFJ3", b"rh_LFJ2",
                  b"rh_THJ5", b"rh_THJ4", b"rh_THJ3", b"rh_THJ2", b"rh_THJ1",
                  b"rh_WRJ2", b"rh_WRJ1"]

COUPLED_JOINTS = {b"rh_FFJ1": b"rh_FFJ2",
                  b"rh_MFJ1": b"rh_MFJ2",
                  b"rh_RFJ1": b"rh_RFJ2",
                  b"rh_LFJ1": b"rh_LFJ2"}

PALM_LINK = [b"rh_palm"]

INITIAL_POS = {b"rh_WRJ2": -0.05866723135113716,
               b"rh_WRJ1": 0.08598895370960236,

               b"rh_FFJ4": -0.05925952824065458,
               b"rh_FFJ3": 0.0,
               b"rh_FFJ2": 0.5306965075027753,
               b"rh_FFJ1": 0.5306965075027753,

               b"rh_MFJ4": 0.015051404275727428,
               b"rh_MFJ3": 0.0,
               b"rh_MFJ2": 0.5364634589883859,
               b"rh_MFJ1": 0.5364634589883859,

               b"rh_RFJ4": -0.056137955514170744,
               b"rh_RFJ3": 0.0,
               b"rh_RFJ2": 0.5362351077308591,
               b"rh_RFJ1": 0.5362351077308591,

               b"rh_LFJ5": 0.0,
               b"rh_LFJ4": -0.216215152247765,
               b"rh_LFJ3": 0.0,
               b"rh_LFJ2": 0.542813974505131,
               b"rh_LFJ1": 0.542813974505131,

               b"rh_THJ5": 1.047,
               b"rh_THJ4": 0.4912634677627796,
               b"rh_THJ3": 0.209,
               b"rh_THJ2": -0.024347361541391634,
               b"rh_THJ1": 0.28372550178530886}

GOAL_ORIENTATION = [0.] * 3


def goal_distance(goal_a, goal_b):
    """Distance to goal.

    :param goal_a: (numpy.array) Achieved_goal (position, orientaiton).
    :param goal_b: (numpy.array) Desired_goal (position, orientaiton).

    :return: (float) Distance to goal.
    """

    goal_a = goal_a[3:]
    goal_b = goal_b[3:]

    if goal_a.shape[0] == 6:
        goal_a = continuous6D_to_quaternion(goal_a)
        goal_b = continuous6D_to_quaternion(goal_b)

    goal_a = np.array(pybullet.getEulerFromQuaternion(goal_a))
    goal_b = np.array(pybullet.getEulerFromQuaternion(goal_b))

    # Ignore y orientation
    #    goal_a = np.delete(np.array(goal_a), 1)
    #    goal_b = np.delete(np.array(goal_b), 1)

    # Account double sided block/cube
    #    if goal_a[0] > np.pi / 2.:
    #        goal_a[0] = np.pi - goal_a[0]
    #    if goal_a[0] < -np.pi / 2.:
    #        goal_a[0] = -np.pi - goal_a[0]
    #    if goal_a[1] > np.pi / 2.:
    #        goal_a[1] = np.pi - goal_a[1]
    #    if goal_a[1] < -np.pi / 2.:
    #        goal_a[1] = -np.pi - goal_a[1]

    #    return np.sum(np.abs(goal_a - goal_b))

    goal_a[1] = goal_b[1]
    goal_a = rotations.euler2quat(goal_a)
    goal_b = rotations.euler2quat(goal_b)

    quaternion_diff = rotations.quat_mul(goal_a, rotations.quat_conjugate(goal_b))
    angle_diff = 2 * np.arccos(np.clip(quaternion_diff[..., 0], -1., 1.))

    return angle_diff


def quaternion_to_continuous6D(orientation):
    """Mapping from SO(3) to 6D representation.
    Zhou et al. "On the Continuity of Rotation Representations in Neural Networks", arXiv:1812.07035v4.

    :param orientation: (list) Quaternion.

    :return: (list) 6D representation of SO(3).
    """
    rot_mat = pybullet.getMatrixFromQuaternion(orientation)
    return rot_mat[:6]


def continuous6D_to_quaternion(orientation):
    """Mapping from 6D representation to SO(3).
    Zhou et al. "On the Continuity of Rotation Representations in Neural Networks", arXiv:1812.07035v4.

    :param orientation: (numpy.array) 6D representation of SO(3).

    :return: (numpy.array) Quaternion.
    """
    b1 = orientation[:3] / np.linalg.norm(orientation[:3])
    b2_ = orientation[3:] - np.dot(b1, orientation[3:]) * b1
    b2 = b2_ / np.linalg.norm(b2_)
    b3 = np.cross(b1, b2)

    return R.from_matrix(np.array([b1, b2, b3])).as_quat()


class ShadowHandManipulateBlockEnv(PyBulletEnv, utils.EzPickle):
    """Shadow Hand Manipulate environment.

    :param orientation_threshold: (float) Threshold to be used.
    :param reward_type: (str) Reward type (dense or sparse).
    :param couple_factor: (list[float * 4]) Joint coupling between Distal and Intermediate phalanges joint (Range: 0-1).
    :param object_path: (str) Path of object to manipulate.
    :param block_length: (float) Block length in metres.
    :param model_path: (str) Path of simulation model.
    :param initial_pos: (dict) Initial model position, {"joint_name": position}.
    :param sim_time_step: (int) Time step to simulate.
    :param sim_frames_skip: (int) How many frames should be skipped.
    :param sim_n_sub_steps: (int) Sub-steps to be taken.
    :param sim_self_collision: (PyBullet.flag) Collision used in model.
    :param  render: (bool) Should render or not.
    :param render_options: (PyBullet.flag) Render options for PyBullet.
    """

    def __init__(self,
                 orientation_threshold=0.1,
                 reward_type="sparse",
                 orientation_type="quaternions",
                 max_steps_per_episode=99,
                 position_gain=0.02,
                 couple_factor=None,
                 object_path=BLOCK_PATH,
                 object_texture_path=BLOCK_TEXTURE_PATH,
                 model_path=MODEL_PATH,
                 block_length=BLOCK_LENGTH,
                 initial_pos=INITIAL_POS,
                 sim_time_step=1.0/240.0,
                 sim_frames_skip=10,
                 sim_n_sub_steps=1,
                 sim_self_collision=pybullet.URDF_USE_SELF_COLLISION,
                 render=False,
                 render_options=None):

        assert reward_type in ["sparse", "dense"], "reward type must be 'sparse' or 'dense'"
        self.reward_type = reward_type

        assert orientation_type in ["quaternions", "6D"], "orientation type must be 'quaternions' or '6D'"
        self.orientation_type = orientation_type

        assert block_length > 0, "Block length must be greater than 0"
        self.block_length = block_length / 2

        self.current_episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode

        self.position_gain = position_gain

        super(ShadowHandManipulateBlockEnv, self).__init__(model_path=model_path,
                                                           initial_pos=initial_pos,
                                                           sim_time_step=sim_time_step,
                                                           sim_frames_skip=sim_frames_skip,
                                                           sim_n_sub_steps=sim_n_sub_steps,
                                                           sim_self_collision=sim_self_collision,
                                                           render=render,
                                                           render_options=render_options)

        utils.EzPickle.__init__(**locals())

        # Joint coupling
        if couple_factor is None:
            self.couple_factor = np.array([1.] * len(COUPLED_JOINTS))
        else:
            self.couple_factor = couple_factor

        # Base position and orientation
        self.base_start_pos = [0.] * 3
        self.base_start_orientation = pybullet.getQuaternionFromEuler([0.] * 3)

        self.orientation_threshold = orientation_threshold

        # Object path
        if object_path.startswith("/"):
            full_path = object_path
        else:
            full_path = os.path.join(os.path.dirname(__file__), "assets", "obj", object_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError("File {} does not exist".format(full_path))

        self.object_path = full_path
        self.object_id = None

        # Object texture
        if object_texture_path.startswith("/"):
            full_path = object_texture_path
        else:
            full_path = os.path.join(os.path.dirname(__file__), "assets", "materials", "textures", object_texture_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError("File {} does not exist".format(full_path))

        self.object_texture_path = full_path

        self.palm_pos = None

    def set_action_space(self):
        """Set action space.

        Iterate over all available joints to determine the count.
        """
        n_actions = 0

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                n_actions += 1

        action_space = spaces.Box(low=-1., high=1., shape=(n_actions,), dtype=np.float64)
        return action_space

    def set_observation_space(self):
        """Set observation space."""
        observation = self.get_observation()
        n_states = len(observation)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_states,), dtype=np.float64)
        return observation_space

    def reset_simulation(self):
        """Reset simulation.

        Reset itself is done in parent class.
        Load all necessary models and set start positions.
        """
        # Set gravity and time-step
        self.physics_client.setGravity(0., 9.81, 0.)
        self.physics_client.setTimeStep(self.sim_time_step)

        # Load robot-model
        if self.sim_self_collision:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation,
                                                         flags=pybullet.URDF_USE_SELF_COLLISION)
        else:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation)

        self.set_initial_pos()

        self.palm_pos = self.get_palm_position()

        # Load object-model
        object_start_position = self.palm_pos + np.array([0., -0.05, 0.06])
        angle = self.np_random.uniform(-np.pi, np.pi)
        axis = self.np_random.uniform(-1., 1., size=3)
        object_start_orientation = pybullet.getQuaternionFromAxisAngle(axis, angle)

        visual_shape_id = self.physics_client.createVisualShape(fileName=self.object_path,
                                                                shapeType=pybullet.GEOM_MESH,
                                                                rgbaColor=None,
                                                                meshScale=[self.block_length] * 3)

        collision_shape_id = self.physics_client.createCollisionShape(fileName=self.object_path,
                                                                      shapeType=pybullet.GEOM_MESH,
                                                                      meshScale=[self.block_length] * 3)

        texture_id = self.physics_client.loadTexture(self.object_texture_path)

        self.object_id = self.physics_client.createMultiBody(baseMass=0.1,
                                                             baseVisualShapeIndex=visual_shape_id,
                                                             baseCollisionShapeIndex=collision_shape_id,
                                                             basePosition=object_start_position.tolist(),
                                                             baseOrientation=object_start_orientation)

        self.physics_client.changeVisualShape(self.object_id, -1, textureUniqueId=texture_id)

        self.goal = self.sample_goal()

        self.current_episode_steps = 0

        observation = self.get_observation()
        return observation

    def get_palm_position(self):
        """Get current palm position."""
        palm_pos = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[12] in PALM_LINK:
                link_state = self.physics_client.getLinkState(self.model_id, n)
                palm_pos.append(link_state[0])

        return np.array(palm_pos[0]).copy()

    def get_observation(self):
        """Get observations.

        Iterate over all movable joints and get the positions and velocities.
        """
        robot_joint_pos = []
        robot_joint_vel = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                joint_state = self.physics_client.getJointState(self.model_id, n)
                robot_joint_pos.append(joint_state[0])
                robot_joint_vel.append(joint_state[1])

        cur_pos, cur_orientation, cur_lin_vel, cur_ang_vel = self.get_current_object_state()

        observation = np.concatenate([robot_joint_pos,
                                      robot_joint_vel,
                                      cur_lin_vel,
                                      cur_ang_vel,
                                      cur_pos,
                                      cur_orientation,
                                      self.goal.copy()])

        return observation.copy()

    def step(self, action):
        """Perform (a) simulation step(s).

        Move movable joints within their range (-1, 1).
        Do the actual simulation.
        Calculate the environment stuff (observation, reward, done, info).
        """
        # Clip action values
        action = np.clip(action, self.action_space.low, self.action_space.high)

        joint_limit_low = []
        joint_limit_high = []
        joints_movable = []
        joints_coupled = []

        # Get joint limits and distinct between movable and coupled joints
        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                joint_limit_low.append(joint_info[8])
                joint_limit_high.append(joint_info[9])
                joints_movable.append(n)
            elif joint_info[1] in COUPLED_JOINTS:
                joints_coupled.append(n)

        joint_limit_low = np.array(joint_limit_low)
        joint_limit_high = np.array(joint_limit_high)

        act_range = (joint_limit_high - joint_limit_low) / 2.
        act_center = (joint_limit_high + joint_limit_low) / 2.

        # Calculate the control action
        ctrl = act_center + act_range * action
        ctrl = np.clip(ctrl, joint_limit_low, joint_limit_high)

        # Actually move the joints
        for n in range(self.n_model_joints):
            if n in joints_movable:
                k = joints_movable.index(n)
                self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                          jointIndex=joints_movable[k],
                                                          controlMode=pybullet.POSITION_CONTROL,
                                                          targetPosition=ctrl[k],
                                                          positionGain=self.position_gain)
            else:
                if n in joints_coupled and n - 1 in joints_movable:
                    k = joints_movable.index(n - 1)
                    self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                              jointIndex=n,
                                                              controlMode=pybullet.POSITION_CONTROL,
                                                              targetPosition=ctrl[k] * self.couple_factor[
                                                                  joints_coupled.index(n)],
                                                              positionGain=self.position_gain)
                else:
                    self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                              jointIndex=n,
                                                              controlMode=pybullet.POSITION_CONTROL,
                                                              targetPosition=0.,
                                                              positionGain=self.position_gain)

        self.do_simulation()

        cur_pos, cur_orientation, _, _ = self.get_current_object_state()
        goal = self.goal.copy()

        observation = self.get_observation()
        cur_state = np.concatenate([cur_pos, cur_orientation])
        done = self.is_success(cur_state, goal)
        info = {"is_success": done}

        reward = self.compute_reward(cur_state, goal, info)

        # Always set done to false to achieve stable end-position
        # Only change it if max episode steps are reached OR block fell off
        done = False

        if not done and self.current_episode_steps == self.max_steps_per_episode:
            done = True

        # Block fell off
        palm_pos = self.get_palm_position()
        if cur_pos[1] > palm_pos[1] + 0.04:
            # Huge negative reward for this action - Nope
            # reward -= 10000.
            # done = True
            info["is_success"] = False

        self.current_episode_steps += 1

        return observation, reward, done, info

    def set_initial_pos(self):
        """Set initial position."""
        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS or joint_info[1] in COUPLED_JOINTS:
                self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                          jointIndex=n,
                                                          controlMode=pybullet.POSITION_CONTROL,
                                                          targetPosition=self.initial_pos[joint_info[1]],
                                                          positionGain=self.position_gain)

        # Settle in
        for _ in range(20):
            self.do_simulation()

    def sample_goal(self):
        """Set goal."""
        orientation = pybullet.getQuaternionFromEuler(GOAL_ORIENTATION)

        if self.orientation_type == "6D":
            orientation = quaternion_to_continuous6D(orientation)

        object_pos, _, _, _ = self.get_current_object_state()

        return np.concatenate([object_pos, np.array(orientation)]).copy()

    def is_success(self, achieved_goal, desired_goal):
        """Goal distance.

        Distance between achieved_goal (current orientation) and goal.
        """
        # Orientation
        distance = goal_distance(achieved_goal, desired_goal)

        return (distance < self.orientation_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward.

        Chose between dense and sparse.
        """
        if self.reward_type == "sparse":
            return self.is_success(achieved_goal, desired_goal) - 1
        else:
            distance = goal_distance(achieved_goal, desired_goal)
            return -distance

    def get_current_object_state(self):
        """Get position and rotation of the block."""
        try:
            object_position, object_orientation = self.physics_client.getBasePositionAndOrientation(self.object_id)
            object_velocity = self.physics_client.getBaseVelocity(self.object_id)
            object_linear_velocity = object_velocity[0]
            object_angular_velocity = object_velocity[1]
        except AttributeError:
            object_position = [0.] * 3
            object_orientation = pybullet.getQuaternionFromEuler([0.] * 3)
            object_linear_velocity = [0.] * 3
            object_angular_velocity = [0.] * 3

        if self.orientation_type == "6D":
            object_orientation = quaternion_to_continuous6D(object_orientation)

        return np.array(object_position).copy(), np.array(object_orientation).copy(), \
               np.array(object_linear_velocity).copy(), np.array(object_angular_velocity).copy()

    def render(self, mode="human", close=False):
        if mode == "rgb_array":
            # Camera defaults
            camera_view_matrix = pybullet.computeViewMatrix(cameraEyePosition=[0.4, -0.35, 0.5],
                                                            cameraTargetPosition=[0., 0., 0.3],
                                                            cameraUpVector=[-1., 0., -1.])
            camera_projection_matrix = pybullet.computeProjectionMatrixFOV(fov=45., aspect=1., nearVal=0.1,
                                                                           farVal=1.1)

            img = self.physics_client.getCameraImage(width=512, height=512,
                                                     viewMatrix=camera_view_matrix,
                                                     projectionMatrix=camera_projection_matrix)

            return img[2]
        else:
            pass


class ShadowHandManipulateBlockGoalEnv(PyBulletGoalEnv, utils.EzPickle):
    """Shadow Hand Manipulate goal environment.

        Used for HER environments.

    :param orientation_threshold: (float) Threshold to be used.
    :param reward_type: (str) Reward type (dense or sparse).
    :param couple_factor: (list[float * 4]) Joint coupling between Distal and Intermediate phalanges joint (Range: 0-1).
    :param object_path: (str) Path of object to manipulate.
    :param model_path: (str) Path of simulation model.
    :param block_length: (float) Block length in metres.
    :param initial_pos: (dict) Initial model position, {"joint_name": position}.
    :param sim_time_step: (int) Time step to simulate.
    :param sim_frames_skip: (int) How many frames should be skipped.
    :param sim_n_sub_steps: (int) Sub-steps to be taken.
    :param sim_self_collision: (PyBullet.flag) Collision used in model.
    :param  render: (bool) Should render or not.
    :param render_options: (PyBullet.flag) Render options for PyBullet.
    """

    def __init__(self,
                 orientation_threshold=0.1,
                 reward_type="sparse",
                 orientation_type="quaternions",
                 max_steps_per_episode=99,
                 position_gain=0.02,
                 couple_factor=None,
                 object_path=BLOCK_PATH,
                 object_texture_path=BLOCK_TEXTURE_PATH,
                 model_path=MODEL_PATH,
                 block_length=BLOCK_LENGTH,
                 initial_pos=INITIAL_POS,
                 sim_time_step=1.0/240.0,
                 sim_frames_skip=10,
                 sim_n_sub_steps=1,
                 sim_self_collision=pybullet.URDF_USE_SELF_COLLISION,
                 render=False,
                 render_options=None):

        assert reward_type in ["sparse", "dense"], "reward type must be 'sparse' or 'dense'"
        self.reward_type = reward_type

        assert orientation_type in ["quaternions", "6D"], "orientation type must be 'quaternions' or '6D'"
        self.orientation_type = orientation_type

        assert block_length > 0, "Block length must be greater than 0"
        self.block_length = block_length / 2

        self.current_episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode

        self.position_gain = position_gain

        super(ShadowHandManipulateBlockGoalEnv, self).__init__(model_path=model_path,
                                                               initial_pos=initial_pos,
                                                               sim_time_step=sim_time_step,
                                                               sim_frames_skip=sim_frames_skip,
                                                               sim_n_sub_steps=sim_n_sub_steps,
                                                               sim_self_collision=sim_self_collision,
                                                               render=render,
                                                               render_options=render_options)

        utils.EzPickle.__init__(**locals())

        # Joint coupling
        if couple_factor is None:
            self.couple_factor = np.array([1.] * len(COUPLED_JOINTS))
        else:
            self.couple_factor = couple_factor

        # Base position and orientation
        self.base_start_pos = [0.] * 3
        self.base_start_orientation = pybullet.getQuaternionFromEuler([0.] * 3)

        self.orientation_threshold = orientation_threshold

        # Object path
        if object_path.startswith("/"):
            full_path = object_path
        else:
            full_path = os.path.join(os.path.dirname(__file__), "assets", "obj", object_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError("File {} does not exist".format(full_path))

        self.object_path = full_path
        self.object_id = None

        # Object texture
        if object_texture_path.startswith("/"):
            full_path = object_texture_path
        else:
            full_path = os.path.join(os.path.dirname(__file__), "assets", "materials", "textures", object_texture_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError("File {} does not exist".format(full_path))

        self.object_texture_path = full_path

        self.palm_pos = None

    def set_action_space(self):
        """Set action space.

        Iterate over all available joints to determine the count.
        """
        n_actions = 0

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                n_actions += 1

        action_space = spaces.Box(low=-1., high=1., shape=(n_actions,), dtype=np.float64)
        return action_space

    def set_observation_space(self):
        """Set observation space.

        Note: HER style.
        """
        observation = self.get_observation()
        observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=observation["achieved_goal"].shape, dtype=np.float64),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=observation["achieved_goal"].shape, dtype=np.float64),
            observation=spaces.Box(-np.inf, np.inf, shape=observation["observation"].shape, dtype=np.float64)
        ))

        return observation_space

    def reset_simulation(self):
        """Reset simulation.

        Reset itself is done in parent class.
        Load all necessary models and set start positions.
        """
        # Set gravity and time-step
        self.physics_client.setGravity(0., 9.81, 0.)
        self.physics_client.setTimeStep(self.sim_time_step)

        # Load robot-model
        if self.sim_self_collision:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation,
                                                         flags=pybullet.URDF_USE_SELF_COLLISION)
        else:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation)

        self.set_initial_pos()

        self.palm_pos = self.get_palm_position()

        # Load object-model
        object_start_position = self.palm_pos + np.array([0., -0.05, 0.06])
        angle = self.np_random.uniform(-np.pi, np.pi)
        axis = self.np_random.uniform(-1., 1., size=3)
        object_start_orientation = pybullet.getQuaternionFromAxisAngle(axis, angle)

        visual_shape_id = self.physics_client.createVisualShape(fileName=self.object_path,
                                                                shapeType=pybullet.GEOM_MESH,
                                                                rgbaColor=None,
                                                                meshScale=[self.block_length] * 3)

        collision_shape_id = self.physics_client.createCollisionShape(fileName=self.object_path,
                                                                      shapeType=pybullet.GEOM_MESH,
                                                                      meshScale=[self.block_length] * 3)

        texture_id = self.physics_client.loadTexture(self.object_texture_path)

        self.object_id = self.physics_client.createMultiBody(baseMass=0.1,
                                                             baseVisualShapeIndex=visual_shape_id,
                                                             baseCollisionShapeIndex=collision_shape_id,
                                                             basePosition=object_start_position.tolist(),
                                                             baseOrientation=object_start_orientation)

        self.physics_client.changeVisualShape(self.object_id, -1, textureUniqueId=texture_id)

        self.goal = self.sample_goal()

        self.current_episode_steps = 0

        observation = self.get_observation()
        return observation

    def get_palm_position(self):
        palm_pos = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[12] in PALM_LINK:
                link_state = self.physics_client.getLinkState(self.model_id, n)
                palm_pos.append(link_state[0])

        return np.array(palm_pos[0]).copy()

    def get_observation(self):
        """Get observations.

        Iterate over all movable joints and get the positions and velocities.

        Note: HER style.
        """
        robot_joint_pos = []
        robot_joint_vel = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                joint_state = self.physics_client.getJointState(self.model_id, n)
                robot_joint_pos.append(joint_state[0])
                robot_joint_vel.append(joint_state[1])

        cur_pos, cur_orientation, cur_lin_vel, cur_ang_vel = self.get_current_object_state()
        achieved_goal = np.concatenate([cur_pos, cur_orientation])
        observation = np.concatenate([robot_joint_pos,
                                      robot_joint_vel,
                                      cur_lin_vel,
                                      cur_ang_vel,
                                      achieved_goal])

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy()
        }

    def step(self, action):
        """Perform (a) simulation step(s).

        Move movable joints within their range (-1, 1).
        Do the actual simulation.
        Calculate the environment stuff (observation, reward, done, info).
        """
        # Clip action values
        action = np.clip(action, self.action_space.low, self.action_space.high)

        joint_limit_low = []
        joint_limit_high = []
        joints_movable = []
        joints_coupled = []

        # Get joint limits and distinct between movable and coupled joints
        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                joint_limit_low.append(joint_info[8])
                joint_limit_high.append(joint_info[9])
                joints_movable.append(n)
            elif joint_info[1] in COUPLED_JOINTS:
                joints_coupled.append(n)

        joint_limit_low = np.array(joint_limit_low)
        joint_limit_high = np.array(joint_limit_high)

        act_range = (joint_limit_high - joint_limit_low) / 2.
        act_center = (joint_limit_high + joint_limit_low) / 2.

        # Calculate the control action
        ctrl = act_center + act_range * action
        ctrl = np.clip(ctrl, joint_limit_low, joint_limit_high)

        # Actually move the joints
        for n in range(self.n_model_joints):
            if n in joints_movable:
                k = joints_movable.index(n)
                self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                          jointIndex=joints_movable[k],
                                                          controlMode=pybullet.POSITION_CONTROL,
                                                          targetPosition=ctrl[k],
                                                          positionGain=self.position_gain)
            else:
                if n in joints_coupled and n - 1 in joints_movable:
                    k = joints_movable.index(n - 1)
                    self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                              jointIndex=n,
                                                              controlMode=pybullet.POSITION_CONTROL,
                                                              targetPosition=ctrl[k] * self.couple_factor[
                                                                  joints_coupled.index(n)],
                                                              positionGain=self.position_gain)
                else:
                    self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                              jointIndex=n,
                                                              controlMode=pybullet.POSITION_CONTROL,
                                                              targetPosition=0.,
                                                              positionGain=self.position_gain)

        self.do_simulation()

        observation = self.get_observation()

        done = self.is_success(observation["achieved_goal"], observation["desired_goal"])
        info = {"is_success": done}
        reward = self.compute_reward(observation["achieved_goal"], observation["desired_goal"], info)

        # Always set done to false to achieve stable end-position
        # Only change it if max episode steps are reached OR block fell off
        done = False

        if not done and self.current_episode_steps == self.max_steps_per_episode:
            done = True

        # Block fell off
        cur_pos = observation["achieved_goal"][:3]
        palm_pos = self.get_palm_position()
        if cur_pos[1] > palm_pos[1] + 0.04:
            # Huge negative reward for this action - Nope
            # reward -= 10000.
            # done = True
            info["is_success"] = False

        self.current_episode_steps += 1

        return observation, reward, done, info

    def set_initial_pos(self):
        """Set initial position."""
        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS or joint_info[1] in COUPLED_JOINTS:
                self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                          jointIndex=n,
                                                          controlMode=pybullet.POSITION_CONTROL,
                                                          targetPosition=self.initial_pos[joint_info[1]],
                                                          positionGain=self.position_gain)

        # Settle in
        for _ in range(20):
            self.do_simulation()

    def sample_goal(self):
        """Set goal."""
        orientation = pybullet.getQuaternionFromEuler(GOAL_ORIENTATION)

        if self.orientation_type == "6D":
            orientation = quaternion_to_continuous6D(orientation)

        object_pos, _, _, _ = self.get_current_object_state()

        return np.concatenate([object_pos, np.array(orientation)]).copy()

    def is_success(self, achieved_goal, desired_goal):
        """Goal distance.

        Distance between achieved_goal (current orientation) and goal.
        """
        distance = goal_distance(achieved_goal, desired_goal)

        return (distance < self.orientation_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward.

        Chose between dense and sparse.
        """
        if self.reward_type == "sparse":
            return self.is_success(achieved_goal, desired_goal) - 1
        else:
            distance = goal_distance(achieved_goal, desired_goal)
            return -distance

    def get_current_object_state(self):
        """Get position and rotation of the block."""
        try:
            object_position, object_orientation = self.physics_client.getBasePositionAndOrientation(self.object_id)
            object_velocity = self.physics_client.getBaseVelocity(self.object_id)
            object_linear_velocity = object_velocity[0]
            object_angular_velocity = object_velocity[1]
        except AttributeError:
            object_position = [0.] * 3
            object_orientation = pybullet.getQuaternionFromEuler([0.] * 3)
            object_linear_velocity = [0.] * 3
            object_angular_velocity = [0.] * 3

        if self.orientation_type == "6D":
            object_orientation = quaternion_to_continuous6D(object_orientation)

        return np.array(object_position).copy(), np.array(object_orientation).copy(), \
               np.array(object_linear_velocity).copy(), np.array(object_angular_velocity).copy()

    def render(self, mode="human", close=False):
        if mode == "rgb_array":
            # Camera defaults
            camera_view_matrix = pybullet.computeViewMatrix(cameraEyePosition=[0.4, -0.35, 0.5],
                                                            cameraTargetPosition=[0., 0., 0.3],
                                                            cameraUpVector=[-1., 0., -1.])
            camera_projection_matrix = pybullet.computeProjectionMatrixFOV(fov=45., aspect=1., nearVal=0.1,
                                                                           farVal=1.1)

            img = self.physics_client.getCameraImage(width=512, height=512,
                                                     viewMatrix=camera_view_matrix,
                                                     projectionMatrix=camera_projection_matrix)

            return img[2]
        else:
            pass
