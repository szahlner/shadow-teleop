import os
import gym
from gym.utils import seeding
import pybullet
from gym_shadow_hand.envs.pybullet_client import PyBulletClient


class PyBulletEnv(gym.Env):
    """Superclass for PyBullet environments.

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
                 model_path,
                 initial_pos,
                 sim_time_step,
                 sim_frames_skip,
                 sim_n_sub_steps,
                 sim_self_collision,
                 render,
                 render_options=None):
        # Model path
        if model_path.startswith("/"):
            full_path = model_path
        else:
            full_path = os.path.join(os.path.dirname(__file__), "assets", "urdf", model_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError("File {} does not exist".format(full_path))

        self.model_path = full_path

        # Simulation settings
        self.sim_time_step = sim_time_step
        self.sim_frames_skip = sim_frames_skip
        self.sim_self_collision = sim_self_collision
        self.sim_n_sub_steps = sim_n_sub_steps

        # Render
        self.visualize = render
        if self.visualize:
            if render_options is None:
                self.physics_client = PyBulletClient(connection_mode=pybullet.GUI)
            else:
                self.physics_client = PyBulletClient(connection_mode=pybullet.GUI, options=render_options)
            self.physics_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = PyBulletClient()

        # Load model to set spaces
        self.model_id = self.physics_client.loadURDF(fileName=self.model_path)
        self.n_model_joints = self.physics_client.getNumJoints(self.model_id)

        # Initial position
        self.initial_pos = initial_pos
        self.set_initial_pos()

        # Seed
        self.seed()

        # Goal
        self.goal = self.sample_goal()

        # Spaces
        self.action_space = self.set_action_space()
        self.observation_space = self.set_observation_space()

    def __del__(self):
        self.close()

    def close(self):
        try:
            self.physics_client.disconnect()
        except pybullet.error:
            pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def do_simulation(self):
        if self.sim_frames_skip == 0:
            self.physics_client.stepSimulation()
        else:
            for _ in range(self.sim_frames_skip):
                self.physics_client.stepSimulation()

    def reset(self):
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(numSubSteps=self.sim_n_sub_steps)
        observation = self.reset_simulation()
        return observation

    def render(self, mode="human", close=False):
        """Rendering is done by PyBullet."""
        pass


    # Methods to override
    # ----------------------------------------------

    def set_initial_pos(self):
        """Set initial position."""
        raise NotImplementedError()

    def sample_goal(self):
        """Sample goal."""
        raise NotImplementedError()

    def set_action_space(self):
        """Set action space."""
        raise NotImplementedError()

    def set_observation_space(self):
        """Set action space."""
        raise NotImplementedError()

    def reset_simulation(self):
        """Resets the simulation."""
        raise NotImplementedError()

    def step(self, action):
        """Perform a step with the given action."""
        raise NotImplementedError()

    def get_observation(self):
        """Get observation."""
        raise NotImplementedError()


class PyBulletGoalEnv(gym.GoalEnv):
    """Superclass for PyBullet goal environments.

    Used for HER environments.

    :param model_path: (str) Path of simulation model.
    :param initial_pos: (dict) Initial model position, {"joint_name": position}.
    :param sim_time_step: (int) Time step to simulate.
    :param sim_frames_skip: (int) How many frames should be skipped.
    :param sim_n_sub_steps: (int) Sub-steps to be taken.
    :param sim_self_collision: (PyBullet flag) Collision used in model.
    :param  render: (bool) Should render or not.
    :param render_options: (optional) Render options for PyBullet.
    """

    def __init__(self,
                 model_path,
                 initial_pos,
                 sim_time_step,
                 sim_frames_skip,
                 sim_n_sub_steps,
                 sim_self_collision,
                 render,
                 render_options=None):
        # Model path
        if model_path.startswith("/"):
            full_path = model_path
        else:
            full_path = os.path.join(os.path.dirname(__file__), "assets", "urdf", model_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError("File {} does not exist".format(full_path))

        self.model_path = full_path

        # Simulation settings
        self.sim_time_step = sim_time_step
        self.sim_frames_skip = sim_frames_skip
        self.sim_self_collision = sim_self_collision
        self.sim_n_sub_steps = sim_n_sub_steps

        # Render
        self.visualize = render
        if self.visualize:
            if render_options is None:
                self.physics_client = PyBulletClient(connection_mode=pybullet.GUI)
            else:
                self.physics_client = PyBulletClient(connection_mode=pybullet.GUI, options=render_options)
            self.physics_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = PyBulletClient()

        # Load model to set spaces
        self.model_id = self.physics_client.loadURDF(fileName=self.model_path)
        self.n_model_joints = self.physics_client.getNumJoints(self.model_id)

        # Initial position
        self.initial_pos = initial_pos
        self.set_initial_pos()

        # Seed
        self.seed()

        # Goal
        self.goal = self.sample_goal()

        # Spaces
        self.action_space = self.set_action_space()
        self.observation_space = self.set_observation_space()

    def __del__(self):
        self.close()

    def close(self):
        try:
            self.physics_client.disconnect()
        except pybullet.error:
            pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def do_simulation(self):
        if self.sim_frames_skip == 0:
            self.physics_client.stepSimulation()
        else:
            for _ in range(self.sim_frames_skip):
                self.physics_client.stepSimulation()

    def reset(self):
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(numSubSteps=self.sim_n_sub_steps)
        observation = self.reset_simulation()
        return observation

    def render(self, mode="human", close=False):
        """Rendering is done by PyBullet."""
        pass

    # Methods to override
    # ----------------------------------------------

    def set_initial_pos(self):
        """Set initial position."""
        raise NotImplementedError()

    def sample_goal(self):
        """Sample goal."""
        raise NotImplementedError()

    def set_action_space(self):
        """Set action space."""
        raise NotImplementedError()

    def set_observation_space(self):
        """Set action space."""
        raise NotImplementedError()

    def reset_simulation(self):
        """Resets the simulation."""
        raise NotImplementedError()

    def step(self, action):
        """Perform a step with the given action."""
        raise NotImplementedError()

    def get_observation(self):
        """Get observation."""
        raise NotImplementedError()