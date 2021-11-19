import os
import numpy as np
import pybullet
from gym import spaces
from gym import utils
from gym_shadow_hand.envs.pybullet_env import PyBulletEnv, PyBulletGoalEnv

MODEL_PATH = "shadow_hand_right.urdf"
SPHERE_GREEN_PATH = os.path.join(os.path.dirname(__file__), "assets", "urdf", "sphere_green.urdf")
SPHERE_BLUE_PATH = os.path.join(os.path.dirname(__file__), "assets", "urdf", "sphere_blue.urdf")
SPHERE_RED_PATH = os.path.join(os.path.dirname(__file__), "assets", "urdf", "sphere_red.urdf")
SPHERE_YELLOW_PATH = os.path.join(os.path.dirname(__file__), "assets", "urdf", "sphere_yellow.urdf")
SPHERE_CYAN_PATH = os.path.join(os.path.dirname(__file__), "assets", "urdf", "sphere_cyan.urdf")

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

FINGER_TIPS = [b"rh_FFtip", b"rh_MFtip", b"rh_RFtip", b"rh_LFtip", b"rh_thtip"]

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

GOAL_POSITIOS = [
    # Goal 1 (normal)
    [[0.00737783, - 0.04983524, 0.4151469],  # Index Finger
     [-0.0160899, - 0.0443932, 0.41973359],  # Middle Finger
     [-0.03960472, - 0.04899406, 0.40823004],  # Ring Finger
     [-0.06979136, - 0.04983475, 0.3939393],  # Little Finger
     [0.0163987, - 0.04818256, 0.37106937]],  # Thumb
    # Goal 2 (pistol)
    [[0.00128918, -0.02065529, 0.43758884],
     [-0.02591066, -0.01888607, 0.43820698],
     [-0.02534194, -0.04953822, 0.32209741],
     [-0.04531284, -0.05274596, 0.3107758],
     [0.03739185, -0.03882779, 0.37454418]],
    # Goal 3 (hanging)
    [[0.01039512, -0.08245648, 0.370413],
     [-0.00963706, -0.07992665, 0.36341932],
     [-0.03092836, -0.07912724, 0.35482527],
     [-0.05460698, -0.07967867, 0.35291384],
     [0.02584797, -0.04864619, 0.37061437]],
    # Goal 4 (rock)
    [[0.00581391, -0.05646441, 0.42523561],
     [0.00393033, -0.04919242, 0.32373276],
     [-0.0177887, -0.049753, 0.3172073],
     [-0.06401634, -0.06163007, 0.40934382],
     [0.03822211, -0.05919186, 0.36643599]]]


def goal_distance(goal_a, goal_b):
    """Distance to goal.

    :param goal_a: (numpy.array) Current / achieved position.
    :param goal_b: (numpy.array) Goal to be reached.

    :return: (float) Distance to goal.
    """
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ShadowHandReachEnv(PyBulletEnv, utils.EzPickle):
    """Shadow Hand Reach environment.

    :param distance_threshold: (float) Threshold to be used.
    :param reward_type: (str) Reward type (dense or sparse).
    :param couple_factor: (list[float * 4]) Joint coupling between Distal and Intermediate phalanges joint (Range: 0-1).
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
                 distance_threshold=0.01,
                 reward_type="sparse",
                 max_steps_per_episode=99,
                 position_gain=0.02,
                 couple_factor=None,
                 model_path=MODEL_PATH,
                 initial_pos=INITIAL_POS,
                 sim_time_step=1.0 / 240.0,
                 sim_frames_skip=10,
                 sim_n_sub_steps=1,
                 sim_self_collision=pybullet.URDF_USE_SELF_COLLISION,
                 render=False,
                 render_options=None):

        self.current_episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode

        self.position_gain = position_gain

        super(ShadowHandReachEnv, self).__init__(model_path=model_path,
                                                 initial_pos=initial_pos,
                                                 sim_time_step=sim_time_step,
                                                 sim_frames_skip=sim_frames_skip,
                                                 sim_n_sub_steps=sim_n_sub_steps,
                                                 sim_self_collision=sim_self_collision,
                                                 render=render,
                                                 render_options=render_options)

        utils.EzPickle.__init__(**locals())

        assert reward_type in ["sparse", "dense"], "reward type must be 'sparse' or 'dense'"
        self.reward_type = reward_type

        # Joint coupling
        if couple_factor is None:
            self.couple_factor = np.array([1.] * len(COUPLED_JOINTS))
        else:
            self.couple_factor = couple_factor

        # Base position and orientation
        self.base_start_pos = [0.] * 3
        self.base_start_orientation = pybullet.getQuaternionFromEuler([0.] * 3)

        self.distance_threshold = distance_threshold

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
        self.physics_client.setGravity(0., 0., -9.81)
        self.physics_client.setTimeStep(self.sim_time_step)

        # Load model
        if self.sim_self_collision:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation,
                                                         flags=pybullet.URDF_USE_SELF_COLLISION)
        else:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation)

        self.current_episode_steps = 0

        self.set_initial_pos()
        self.goal = self.sample_goal()

        if self.visualize:
            # Load goal sphere(s) for show
            goal = self.goal.copy()
            self.physics_client.loadURDF(fileName=SPHERE_GREEN_PATH, basePosition=goal[0].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_BLUE_PATH, basePosition=goal[1].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_RED_PATH, basePosition=goal[2].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_YELLOW_PATH, basePosition=goal[3].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_CYAN_PATH, basePosition=goal[4].tolist(), useFixedBase=1)

        observation = self.get_observation()
        return observation

    def get_observation(self):
        """Get observations.

        Iterate over all movable joints and get the positions and velocities.
        """
        joint_pos = []
        joint_vel = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                joint_state = self.physics_client.getJointState(self.model_id, n)
                joint_pos.append(joint_state[0])
                joint_vel.append(joint_state[1])

        cur_pos = self.get_current_position()

        observation = np.concatenate([joint_pos, joint_vel, cur_pos.flatten().copy(), self.goal.flatten().copy()])
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

        cur_pos = self.get_current_position()
        goal = self.goal.copy()

        observation = self.get_observation()
        done = self.is_success(cur_pos.flatten(), goal.flatten())
        info = {"is_success": done}
        reward = self.compute_reward(cur_pos.flatten(), goal.flatten(), info)

        done = False

        if not done and self.current_episode_steps == self.max_steps_per_episode:
            done = True

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
        """Randomly chose goal."""
        choice = self.np_random.choice([n for n in range(len(GOAL_POSITIOS))])
        goal = np.array(GOAL_POSITIOS[choice]).copy()

        if self.np_random.uniform() < 0.1:
            goal = self.get_current_position()

        return goal.copy()

    def is_success(self, achieved_goal, desired_goal):
        """Goal distance.

        Distance between achieved_goal (current position) and goal.
        """
        distance = goal_distance(achieved_goal, desired_goal)
        return (distance < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward.

        Chose between dense and sparse.
        """
        if self.reward_type == "sparse":
            return self.is_success(achieved_goal, desired_goal) - 1
        else:
            distance = goal_distance(achieved_goal, desired_goal)
            return -distance

    def get_current_position(self):
        """Get positions of all fingertips."""
        pos = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in FINGER_TIPS:
                link_state = self.physics_client.getLinkState(self.model_id, n)
                pos.append(link_state[0])

        return np.array(pos).copy()

    def render(self, mode="human", close=False):
        if mode == "rgb_array":
            # Load goal sphere(s) for show
            self.physics_client.loadURDF(fileName=SPHERE_GREEN_PATH, basePosition=self.goal[0].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_BLUE_PATH, basePosition=self.goal[1].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_RED_PATH, basePosition=self.goal[2].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_YELLOW_PATH, basePosition=self.goal[3].tolist(),
                                         useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_CYAN_PATH, basePosition=self.goal[4].tolist(), useFixedBase=1)

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


class ShadowHandReachGoalEnv(PyBulletGoalEnv, utils.EzPickle):
    """Shadow Hand Reach goal environment.

    Used for HER environments.

    :param distance_threshold: (float) Threshold to be used.
    :param reward_type: (str) Reward type (dense or sparse).
    :param couple_factor: (list[float * 4]) Joint coupling between Distal and Intermediate phalanges joint (Range: 0-1).
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
                 distance_threshold=0.01,
                 reward_type="sparse",
                 max_steps_per_episode=99,
                 position_gain=0.02,
                 couple_factor=None,
                 model_path=MODEL_PATH,
                 initial_pos=INITIAL_POS,
                 sim_time_step=1.0 / 240.0,
                 sim_frames_skip=10,
                 sim_n_sub_steps=1,
                 sim_self_collision=pybullet.URDF_USE_SELF_COLLISION,
                 render=False,
                 render_options=None):

        self.current_episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode

        self.position_gain = position_gain

        super(ShadowHandReachGoalEnv, self).__init__(model_path=model_path,
                                                     initial_pos=initial_pos,
                                                     sim_time_step=sim_time_step,
                                                     sim_frames_skip=sim_frames_skip,
                                                     sim_n_sub_steps=sim_n_sub_steps,
                                                     sim_self_collision=sim_self_collision,
                                                     render=render,
                                                     render_options=render_options)

        utils.EzPickle.__init__(**locals())

        assert reward_type in ["sparse", "dense"], "reward type must be 'sparse' or 'dense'"
        self.reward_type = reward_type

        # Joint coupling
        if couple_factor is None:
            self.couple_factor = np.array([1.] * len(COUPLED_JOINTS))
        else:
            self.couple_factor = couple_factor

        self.reward_type = reward_type

        # Base position and orientation
        self.base_start_pos = [0.] * 3
        self.base_start_orientation = pybullet.getQuaternionFromEuler([0.] * 3)

        self.distance_threshold = distance_threshold

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
        self.physics_client.setGravity(0., 0., -9.81)
        self.physics_client.setTimeStep(self.sim_time_step)

        # Load model
        if self.sim_self_collision:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation,
                                                         flags=pybullet.URDF_USE_SELF_COLLISION)
        else:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation)

        self.current_episode_steps = 0

        self.set_initial_pos()
        self.goal = self.sample_goal()

        if self.visualize:
            # Load goal sphere(s) for show
            goal = self.goal.copy()
            self.physics_client.loadURDF(fileName=SPHERE_GREEN_PATH, basePosition=goal[0].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_BLUE_PATH, basePosition=goal[1].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_RED_PATH, basePosition=goal[2].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_YELLOW_PATH, basePosition=goal[3].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_CYAN_PATH, basePosition=goal[4].tolist(), useFixedBase=1)

        observation = self.get_observation()
        return observation

    def get_observation(self):
        """Get observations.

        Iterate over all movable joints and get the positions and velocities.

        Note: HER style.
        """
        joint_pos = []
        joint_vel = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                joint_state = self.physics_client.getJointState(self.model_id, n)
                joint_pos.append(joint_state[0])
                joint_vel.append(joint_state[1])

        cur_pos = self.get_current_position()

        observation = np.concatenate([joint_pos, joint_vel, cur_pos.flatten().copy()])
        return {
            "observation": observation.copy(),
            "achieved_goal": cur_pos.flatten().copy(),
            "desired_goal": self.goal.flatten().copy()
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

        done = False

        if not done and self.current_episode_steps == self.max_steps_per_episode:
            done = True

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
        """Randomly chose goal."""
        choice = self.np_random.choice([n for n in range(len(GOAL_POSITIOS))])
        goal = np.array(GOAL_POSITIOS[choice]).copy()

        if self.np_random.uniform() < 0.1:
            goal = self.get_current_position()

        return goal.copy()

    def is_success(self, achieved_goal, desired_goal):
        """Goal distance.

        Distance between achieved_goal (current position) and goal.
        """
        distance = goal_distance(achieved_goal, desired_goal)
        return (distance < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward.

        Chose between dense and sparse.
        """
        if self.reward_type == "sparse":
            return self.is_success(achieved_goal, desired_goal) - 1
        else:
            distance = goal_distance(achieved_goal, desired_goal)
            return -distance

    def get_current_position(self):
        """Get positions of all fingertips."""
        pos = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in FINGER_TIPS:
                link_state = self.physics_client.getLinkState(self.model_id, n)
                pos.append(link_state[0])

        return np.array(pos).copy()

    def render(self, mode="human", close=False):
        if mode == "rgb_array":
            # Load goal sphere(s) for show
            self.physics_client.loadURDF(fileName=SPHERE_GREEN_PATH, basePosition=self.goal[0].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_BLUE_PATH, basePosition=self.goal[1].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_RED_PATH, basePosition=self.goal[2].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_YELLOW_PATH, basePosition=self.goal[3].tolist(),
                                         useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_CYAN_PATH, basePosition=self.goal[4].tolist(), useFixedBase=1)

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


class ShadowHandReachEnvV0(PyBulletEnv, utils.EzPickle):
    """Shadow Hand Reach environment.
    Legacy environment.

    :param distance_threshold: (float) Threshold to be used.
    :param reward_type: (str) Reward type (dense or sparse).
    :param couple_factor: (list[float * 4]) Joint coupling between Distal and Intermediate phalanges joint (Range: 0-1).
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
                 distance_threshold=0.01,
                 reward_type="dense",
                 max_steps_per_episode=99,
                 position_gain=0.02,
                 couple_factor=None,
                 model_path=MODEL_PATH,
                 initial_pos=INITIAL_POS,
                 sim_time_step=1.0 / 240.0,
                 sim_frames_skip=10,
                 sim_n_sub_steps=1,
                 sim_self_collision=pybullet.URDF_USE_SELF_COLLISION,
                 render=False,
                 render_options=None):

        self.current_episode_steps = 0
        self.max_steps_per_episode = max_steps_per_episode

        self.position_gain = position_gain

        super(ShadowHandReachEnvV0, self).__init__(model_path=model_path,
                                                   initial_pos=initial_pos,
                                                   sim_time_step=sim_time_step,
                                                   sim_frames_skip=sim_frames_skip,
                                                   sim_n_sub_steps=sim_n_sub_steps,
                                                   sim_self_collision=sim_self_collision,
                                                   render=render,
                                                   render_options=render_options)

        utils.EzPickle.__init__(**locals())

        assert reward_type in ["sparse", "dense"], "reward type must be 'sparse' or 'dense'"
        self.reward_type = reward_type

        # Joint coupling
        if couple_factor is None:
            self.couple_factor = np.array([1.] * len(COUPLED_JOINTS))
        else:
            self.couple_factor = couple_factor

        # Base position and orientation
        self.base_start_pos = [0.] * 3
        self.base_start_orientation = pybullet.getQuaternionFromEuler([0.] * 3)

        self.distance_threshold = distance_threshold

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
        self.physics_client.setGravity(0., 0., -9.81)
        self.physics_client.setTimeStep(self.sim_time_step)

        # Load model
        if self.sim_self_collision:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation,
                                                         flags=pybullet.URDF_USE_SELF_COLLISION)
        else:
            self.model_id = self.physics_client.loadURDF(fileName=self.model_path,
                                                         basePosition=self.base_start_pos,
                                                         baseOrientation=self.base_start_orientation)

        self.current_episode_steps = 0

        self.set_initial_pos()
        self.goal = self.sample_goal()

        if self.visualize:
            # Load goal sphere(s) for show
            goal = self.goal.copy()
            self.physics_client.loadURDF(fileName=SPHERE_GREEN_PATH, basePosition=goal[0].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_BLUE_PATH, basePosition=goal[1].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_RED_PATH, basePosition=goal[2].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_YELLOW_PATH, basePosition=goal[3].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_CYAN_PATH, basePosition=goal[4].tolist(), useFixedBase=1)

        observation = self.get_observation()
        return observation

    def get_observation(self):
        """Get observations.

        Iterate over all movable joints and get the positions and velocities.
        """
        joint_pos = []
        joint_vel = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in MOVABLE_JOINTS:
                joint_state = self.physics_client.getJointState(self.model_id, n)
                joint_pos.append(joint_state[0])
                joint_vel.append(joint_state[1])

        cur_pos = self.get_current_position()

        observation = np.concatenate([joint_pos, joint_vel, cur_pos.flatten().copy(), self.goal.flatten().copy()])
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
                                                          positionGain=self.position_gain)  # use default for legacy
            else:
                if n in joints_coupled and n - 1 in joints_movable:
                    k = joints_movable.index(n - 1)
                    self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                              jointIndex=n,
                                                              controlMode=pybullet.POSITION_CONTROL,
                                                              targetPosition=ctrl[k] * self.couple_factor[
                                                                  joints_coupled.index(n)],
                                                              positionGain=self.position_gain)  # use default for legacy
                else:
                    self.physics_client.setJointMotorControl2(bodyUniqueId=self.model_id,
                                                              jointIndex=n,
                                                              controlMode=pybullet.POSITION_CONTROL,
                                                              targetPosition=0.,
                                                              positionGain=self.position_gain)  # use default for legacy

        self.do_simulation()

        cur_pos = self.get_current_position()
        goal = self.goal.copy()

        observation = self.get_observation()
        done = self.is_success(cur_pos.flatten(), goal.flatten())
        info = {"is_success": done}
        reward = self.compute_reward(cur_pos.flatten(), goal.flatten(), info)

        if not done and self.current_episode_steps == self.max_steps_per_episode:
            done = True

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
                                                          positionGain=self.position_gain)  # use default for legacy

        # Settle in
        for _ in range(20):
            self.do_simulation()

    def sample_goal(self):
        """Randomly chose goal."""
        choice = self.np_random.choice([n for n in range(len(GOAL_POSITIOS))])
        goal = np.array(GOAL_POSITIOS[choice]).copy()

        if self.np_random.uniform() < 0.1:
            goal = self.get_current_position()

        return goal.copy()

    def is_success(self, achieved_goal, desired_goal):
        """Goal distance.

        Distance between achieved_goal (current position) and goal.
        """
        distance = goal_distance(achieved_goal, desired_goal)
        return (distance < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute reward.

        Chose between dense and sparse.
        """
        if self.reward_type == "sparse":
            return self.is_success(achieved_goal, desired_goal) - 1
        else:
            distance = goal_distance(achieved_goal, desired_goal)
            return -distance

    def get_current_position(self):
        """Get positions of all fingertips."""
        pos = []

        for n in range(self.n_model_joints):
            joint_info = self.physics_client.getJointInfo(self.model_id, n)

            if joint_info[1] in FINGER_TIPS:
                link_state = self.physics_client.getLinkState(self.model_id, n)
                pos.append(link_state[0])

        return np.array(pos).copy()

    def render(self, mode="human", close=False):
        if mode == "rgb_array":
            # Load goal sphere(s) for show
            self.physics_client.loadURDF(fileName=SPHERE_GREEN_PATH, basePosition=self.goal[0].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_BLUE_PATH, basePosition=self.goal[1].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_RED_PATH, basePosition=self.goal[2].tolist(), useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_YELLOW_PATH, basePosition=self.goal[3].tolist(),
                                         useFixedBase=1)
            self.physics_client.loadURDF(fileName=SPHERE_CYAN_PATH, basePosition=self.goal[4].tolist(), useFixedBase=1)

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
