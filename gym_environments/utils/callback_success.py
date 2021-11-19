import os
import typing
import warnings
from typing import Union, Optional

import gym
import numpy as np

from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.her.her import HERGoalEnvWrapper


def evaluate_success_policy(model, env, n_eval_episodes=10, deterministic=True, render=False):
    """
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.
    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths, episode_successes = [], [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = []
        episode_success = []
        episode_length = 0
        n = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward.append(reward[0])
            episode_length += 1

            try:
                episode_success.append(info["is_success"])
            except TypeError:
                # Account for HER
                episode_success.append(info[0]["is_success"])

            if render:
                env.render()

            n += 1

        try:
            if n < model.env.max_steps_per_episode:
                for k in range(n, model.env.max_steps_per_episode + 1):
                    episode_reward.append(reward[0])

                    try:
                        episode_success.append(info["is_success"])
                    except TypeError:
                        # Account for HER
                        episode_success.append(info[0]["is_success"])
        except AttributeError:
            # Workaround for HER
            pass

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(episode_success)

    return np.array(episode_rewards), np.array(episode_lengths), np.array(episode_successes)


class EvalSuccessCallback(EvalCallback):

    def __init__(self,
                 eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1,
                 rank: int = 0):

        super(EvalSuccessCallback, self).__init__(eval_env=eval_env,
                                                  callback_on_new_best=callback_on_new_best,
                                                  n_eval_episodes=n_eval_episodes,
                                                  eval_freq=eval_freq,
                                                  log_path=log_path,
                                                  best_model_save_path=best_model_save_path,
                                                  deterministic=deterministic,
                                                  render=render,
                                                  verbose=verbose)
        self.evaluations_successes = []
        self.training_episode_successes = []
        self.expert_batch_sizes = []
        self.rank = rank

        self.episodes_eval_freq = 200
        self.episodes_max = 50000
        self.episodes = []
        self.episodes_evaluations_timesteps = []
        self.episodes_evaluations_results = []
        self.episodes_evaluations_length = []
        self.episodes_evaluations_successes = []
        self.episodes_expert_batch_sizes = []
        self.episodes_training_episode_successes = []
        self.episodes_best_mean_reward = -np.inf
        self.episode_current = 1

        try:
            if eval_env.unwrapped.spec.id.endswith("v0"):
                self.legacy_env = True
            else:
                self.legacy_env = False
        except AttributeError:
            # Workaround for HER
            self.legacy_env = False

    def _on_step(self) -> bool:

        cur_episode = len(self.locals["epoch_episode_steps"]) + 1

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.training_episode_successes.append(np.mean(self.locals["episode_successes"][-100:]))

            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths, episode_successes = \
                evaluate_success_policy(self.model, self.eval_env,
                                        n_eval_episodes=self.n_eval_episodes,
                                        render=self.render,
                                        deterministic=self.deterministic)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_success, std_success = np.mean(episode_successes), np.std(episode_successes)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            try:
                expert_use = self.model.expert_use
            except AttributeError:
                expert_use = False

            if expert_use:
                batch_size = (self.model.expert_limit_success - mean_success)
                batch_size *= self.model.expert_batch_size / self.model.expert_limit_success
                batch_size = np.ceil(batch_size).astype(np.int32)
                batch_size = np.max([0, batch_size])
                #self.model.expert_batch_size_current = batch_size
                #batch_size = self.model.expert_batch_size_current
            else:
                batch_size = -1

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_successes.append(episode_successes)
                self.expert_batch_sizes.append(batch_size)
                np.savez_compressed("{}{}".format(self.log_path, self.rank),
                                    timesteps=self.evaluations_timesteps,
                                    results=self.evaluations_results,
                                    ep_lengths=self.evaluations_length,
                                    successes=self.evaluations_successes,
                                    training_successes=self.training_episode_successes,
                                    expert_batch_sizes=self.expert_batch_sizes)

            if self.verbose > 0:
                print("Eval num_timesteps: {}".format(self.num_timesteps))
                print("Episode mean reward: {:.5f} +/- {:.5f}".format(mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
                print("Episode mean success: {:.10f} +/- {:.10f}".format(mean_success, std_success))
                print("Training episode mean success: {:.10f}".format(np.mean(self.locals["episode_successes"][-100:])))
                print("Current demo_batch_size: {}".format(batch_size))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward

            if self.verbose > 0:
                print("")

        if self.episodes_eval_freq > 0 and self.legacy_env and \
                cur_episode > self.episode_current and \
                cur_episode % self.episodes_eval_freq == 0:

            self.episode_current = cur_episode
            self.episodes_training_episode_successes.append(np.mean(self.locals["episode_successes"][-100:]))

            episode_rewards, episode_lengths, episode_successes = \
                evaluate_success_policy(self.model, self.eval_env,
                                        n_eval_episodes=self.n_eval_episodes,
                                        render=self.render,
                                        deterministic=self.deterministic)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_success, std_success = np.mean(episode_successes), np.std(episode_successes)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            expert_use = False

            if expert_use:
                batch_size = (self.model.expert_limit_success - mean_success)
                batch_size *= self.model.expert_batch_size / self.model.expert_limit_success
                batch_size = np.ceil(batch_size).astype(np.int32)
                batch_size = np.max([0, batch_size])
                self.model.expert_batch_size_current = batch_size
            else:
                batch_size = -1

            if self.log_path is not None:
                self.episodes_evaluations_timesteps.append(self.num_timesteps)
                self.episodes_evaluations_results.append(episode_rewards)
                self.episodes_evaluations_length.append(episode_lengths)
                self.episodes_evaluations_successes.append(episode_successes)
                self.episodes_expert_batch_sizes.append(batch_size)
                self.episodes.append(cur_episode)
                np.savez_compressed("{}_episode_{}".format(self.log_path, self.rank),
                                    timesteps=self.episodes_evaluations_timesteps,
                                    results=self.episodes_evaluations_results,
                                    ep_lengths=self.episodes_evaluations_length,
                                    successes=self.episodes_evaluations_successes,
                                    training_successes=self.training_episode_successes,
                                    expert_batch_sizes=self.episodes_expert_batch_sizes,
                                    episodes=self.episodes)

            if self.verbose > 0:
                print("Current episode: {}".format(cur_episode))
                print("Episode mean reward: {:.5f} +/- {:.5f}".format(mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
                print("Episode mean success: {:.10f} +/- {:.10f}".format(mean_success, std_success))
                print("Training episode mean success: {:.10f}".format(np.mean(self.locals["episode_successes"][-100:])))
                print("Current demo_batch_size: {}".format(batch_size))

            if mean_reward > self.episodes_best_mean_reward:
                if self.verbose > 0:
                    print("New best mean episodes reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model_episodes'))
                self.episodes_best_mean_reward = mean_reward

            if self.verbose > 0:
                print("")

            if self.legacy_env and (cur_episode) % self.episodes_max == 0:
                return False

        return True
