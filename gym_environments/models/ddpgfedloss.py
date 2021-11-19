from functools import reduce
import os
import numpy as np
import pickle
import time
from collections import deque

from mpi4py import MPI

import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter
from stable_baselines import DDPG
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.math_util import unscale_action, scale_action
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.mpi_adam import MpiAdam


class DDPGfEDLoss(DDPG):
    """
    Custom version of Deep Deterministic Policy Gradient (DDPG) to use with expert demonstrations.
    Similar to DDPG from Demonstrations (DDPGfD).
    """

    def __init__(self, policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1,
                 expert_use=False, expert_data=None, expert_batch_size=64, lambda1=0.001, lambda2=0.0078):

        self.expert_use = expert_use

        if self.expert_use:
            self.expert_data = expert_data
            self.expert_batch_size = expert_batch_size
            self.expert_batch_size_current = expert_batch_size
            self.expert_limit_success = np.inf
            self.demo_data = None
            self.demo_size = None
            self.expert_loss = None
            self.lambda1 = lambda1
            self.lambda2 = lambda2
        else:
            self.expert_data = None
            self.expert_batch_size = 0
            self.expert_batch_size_current = 0
            self.expert_limit_success = 0

        super(DDPGfEDLoss, self).__init__(policy=policy, env=env, gamma=gamma, memory_policy=memory_policy,
                                          eval_env=eval_env, nb_train_steps=nb_train_steps,
                                          nb_rollout_steps=nb_rollout_steps, nb_eval_steps=nb_eval_steps,
                                          param_noise=param_noise, action_noise=action_noise,
                                          normalize_observations=normalize_observations, tau=tau, batch_size=batch_size,
                                          param_noise_adaption_interval=param_noise_adaption_interval,
                                          normalize_returns=normalize_returns, enable_popart=enable_popart,
                                          observation_range=observation_range, critic_l2_reg=critic_l2_reg,
                                          return_range=return_range, actor_lr=actor_lr, critic_lr=critic_lr,
                                          clip_norm=clip_norm, reward_scale=reward_scale, render=render,
                                          render_eval=render_eval, memory_limit=memory_limit, buffer_size=buffer_size,
                                          random_exploration=random_exploration, verbose=verbose,
                                          tensorboard_log=tensorboard_log, _init_setup_model=_init_setup_model,
                                          policy_kwargs=policy_kwargs, full_tensorboard_log=full_tensorboard_log,
                                          seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if self.expert_use:
            self._init_demo_buffer()

    def get_random_action(self):
        return np.random.uniform(-1.5, 1.5, self.env.action_space.shape[0])

    def _init_demo_buffer(self):
        rank = MPI.COMM_WORLD.Get_rank()

        if rank == 0 and self.verbose >= 1:
            print("Start init demo buffer")

        data_path = "{}{}.npz".format(self.expert_data[:-4], rank)

        if not os.path.exists(data_path):
            import shutil
            shutil.copy(self.expert_data, data_path)

        demo_data = np.load(data_path)
        self.demo_size = len(demo_data.f.actions)
        self.demo_buffer = ReplayBuffer(self.demo_size)

        self.demo_data = {
            "obs": demo_data["obs"].copy(),
            "actions": demo_data["actions"].copy(),
            "rewards": demo_data["rewards"].copy(),
            "episode_starts": demo_data["episode_starts"].copy()
        }

        for n in range(1, self.demo_size):
            obs = self.demo_data["obs"][n - 1]
            self.demo_buffer.add(obs,
                                 self.demo_data["actions"][n],
                                 self.demo_data["rewards"][n] * self.reward_scale,
                                 self.demo_data["obs"][n],
                                 self.demo_data["episode_starts"][n].astype(np.float32))
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs]))

        del demo_data

        os.remove(data_path)
        MPI.COMM_WORLD.Barrier()

        if rank == 0 and self.verbose >= 1:
            print("Done init demo buffer")

    def _setup_actor_optimizer(self):
        """
        setup the optimizer for the actor
        """
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')

        if self.expert_use:
            zeros_batch_normal = np.zeros(self.batch_size - self.expert_batch_size)
            ones_batch_expert = np.ones(self.expert_batch_size)
            mask = np.concatenate((zeros_batch_normal, ones_batch_expert), axis=0)
            mask_main = tf.reshape(tf.boolean_mask(self.critic_tf > self.critic_with_actor_tf, mask), [-1])

            self.expert_loss = tf.reduce_sum(tf.square(
                tf.boolean_mask(tf.boolean_mask(self.actor_tf, mask), mask_main, axis=0) - \
                tf.boolean_mask(tf.boolean_mask(self.actions, mask), mask_main, axis=0)
            ))

            self.actor_loss = -self.lambda1 * tf.reduce_mean(self.critic_with_actor_tf)
            self.actor_loss += self.lambda2 * self.expert_loss
        else:
            self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)

        actor_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        if self.verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf_util.flatgrad(self.actor_loss, tf_util.get_trainable_vars('model/pi/'),
                                            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/pi/'), beta1=0.9, beta2=0.999,
                                       epsilon=1e-08)

    """
    #mask = np.concatenate((np.zeros(self.batch_size - self.demo_batch_size), np.ones(self.demo_batch_size)), axis=0)
    target_Q_pi_tf = self.target.Q_pi_tf
    clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
    target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)  # y = r + gamma*Q(pi)
    self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))  # (y-Q(critic))^2

    if self.bc_loss == 1 and self.q_filter == 1:
        maskMain = tf.reshape(tf.boolean_mask(self.main.Q_tf > self.main.Q_pi_tf, mask), [
            -1])  # where is the demonstrator action better than actor action according to the critic?
        self.cloning_loss_tf = tf.reduce_sum(tf.square(
            tf.boolean_mask(tf.boolean_mask((self.main.pi_tf), mask), maskMain, axis=0) - tf.boolean_mask(
                tf.boolean_mask((batch_tf['u']), mask), maskMain, axis=0)))
        self.pi_loss_tf = -self.lambda1 * tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf

    elif self.bc_loss == 1 and self.q_filter == 0:
        self.cloning_loss_tf = tf.reduce_sum(
            tf.square(tf.boolean_mask((self.main.pi_tf), mask) - tf.boolean_mask((batch_tf['u']), mask)))
        self.pi_loss_tf = -self.lambda1 * tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.lambda1 * self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.pi_loss_tf += self.lambda2 * self.cloning_loss_tf

    else:
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.cloning_loss_tf = tf.reduce_sum(tf.square(self.main.pi_tf - batch_tf['u']))  # random values
    """

    def _train_step(self, step, writer, log=False):
        """
        run a step of training from batch

        :param step: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param log: (bool) whether or not to log to metadata
        :return: (float, float) critic loss, actor loss
        """
        if self.expert_use and self.expert_batch_size > 0:
            # Get a batch
            batch_size = self.batch_size - self.expert_batch_size
            obs, actions, rewards, next_obs, terminals = self.replay_buffer.sample(batch_size=batch_size,
                                                                                   env=self._vec_normalize_env)

            _obs, _actions, _rewards, _next_obs, _terminals = self.demo_buffer.sample(
                batch_size=self.expert_batch_size,
                env=self._vec_normalize_env)

            obs = np.append(obs, _obs, axis=0)
            actions = np.append(actions, _actions, axis=0)
            rewards = np.append(rewards, _rewards, axis=0)
            next_obs = np.append(next_obs, _next_obs, axis=0)
            terminals = np.append(terminals, _terminals, axis=0)
        else:
            # Get a batch
            obs, actions, rewards, next_obs, terminals = self.replay_buffer.sample(batch_size=self.batch_size,
                                                                                   env=self._vec_normalize_env)
        # Reshape to match previous behavior and placeholder shape
        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_q],
                                                        feed_dict={
                                                            self.obs_target: next_obs,
                                                            self.rewards: rewards,
                                                            self.terminals_ph: terminals
                                                        })
            self.ret_rms.update(target_q.flatten())
            self.sess.run(self.renormalize_q_outputs_op, feed_dict={
                self.old_std: np.array([old_std]),
                self.old_mean: np.array([old_mean]),
            })

        else:
            target_q = self.sess.run(self.target_q, feed_dict={
                self.obs_target: next_obs,
                self.rewards: rewards,
                self.terminals_ph: terminals
            })

        # Get all gradients and perform a synced update.
        ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
        td_map = {
            self.obs_train: obs,
            self.actions: actions,
            self.action_train_ph: actions,
            self.rewards: rewards,
            self.critic_target: target_q,
            self.param_noise_stddev: 0 if self.param_noise is None else self.param_noise.current_stddev
        }
        if writer is not None:
            # run loss backprop with summary if the step_id was not already logged (can happen with the right
            # parameters as the step value is only an estimate)
            if self.full_tensorboard_log and log and step not in self.tb_seen_steps:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, actor_grads, actor_loss, critic_grads, critic_loss = \
                    self.sess.run([self.summary] + ops, td_map, options=run_options, run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%d' % step)
                self.tb_seen_steps.append(step)
            else:
                summary, actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run([self.summary] + ops,
                                                                                            td_map)
            writer.add_summary(summary, step)
        else:
            actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, td_map)

        self.actor_optimizer.update(actor_grads, learning_rate=self.actor_lr)
        self.critic_optimizer.update(critic_grads, learning_rate=self.critic_lr)

        return critic_loss, actor_loss

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DDPG",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # a list for tensorboard logging, to prevent logging with the same step number, if it already occured
            self.tb_seen_steps = []

            rank = MPI.COMM_WORLD.Get_rank()

            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            episode_successes = []

            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                obs = self.env.reset()
                # Retrieve unnormalized observation for saving into the buffer
                if self._vec_normalize_env is not None:
                    obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                eval_obs = None
                if self.eval_env is not None:
                    eval_obs = self.eval_env.reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                eval_episode_rewards = []
                eval_qs = []
                epoch_actions = []
                epoch_qs = []
                epoch_episodes = 0
                epoch = 0

                callback.on_training_start(locals(), globals())

                while True:
                    for _ in range(log_interval):
                        callback.on_rollout_start()
                        # Perform rollouts.
                        for _ in range(self.nb_rollout_steps):

                            if total_steps >= total_timesteps:
                                callback.on_training_end()
                                return self

                            # Predict next action.
                            action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                            assert action.shape == self.env.action_space.shape

                            # Execute next action.
                            if rank == 0 and self.render:
                                self.env.render()

                            # Randomly sample actions from a uniform distribution
                            # with a probability self.random_exploration (used in HER + DDPG)
                            if np.random.rand() < self.random_exploration:
                                # actions sampled from action space are from range specific to the environment
                                # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                                unscaled_action = self.action_space.sample()
                                action = scale_action(self.action_space, unscaled_action)
                            else:
                                # inferred actions need to be transformed to environment action_space before stepping
                                unscaled_action = unscale_action(self.action_space, action)

                            new_obs, reward, done, info = self.env.step(unscaled_action)

                            self.num_timesteps += 1

                            if callback.on_step() is False:
                                callback.on_training_end()
                                return self

                            step += 1
                            total_steps += 1
                            if rank == 0 and self.render:
                                self.env.render()

                            # Book-keeping.
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)

                            # Store only the unnormalized version
                            if self._vec_normalize_env is not None:
                                new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                                reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                            else:
                                # Avoid changing the original ones
                                obs_, new_obs_, reward_ = obs, new_obs, reward

                            self._store_transition(obs_, action, reward_, new_obs_, done)
                            obs = new_obs
                            # Save the unnormalized observation
                            if self._vec_normalize_env is not None:
                                obs_ = new_obs_

                            episode_reward += reward_
                            episode_step += 1

                            if writer is not None:
                                ep_rew = np.array([reward_]).reshape((1, -1))
                                ep_done = np.array([done]).reshape((1, -1))
                                tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                                    writer, self.num_timesteps)

                            if done:
                                # Episode done.
                                epoch_episode_rewards.append(episode_reward)
                                episode_rewards_history.append(episode_reward)
                                epoch_episode_steps.append(episode_step)
                                episode_reward = 0.
                                episode_step = 0
                                epoch_episodes += 1
                                episodes += 1

                                maybe_is_success = info.get('is_success')
                                if maybe_is_success is not None:
                                    episode_successes.append(float(maybe_is_success))

                                self._reset()
                                if not isinstance(self.env, VecEnv):
                                    obs = self.env.reset()

                        callback.on_rollout_end()
                        # Train.
                        epoch_actor_losses = []
                        epoch_critic_losses = []
                        epoch_adaptive_distances = []
                        if total_steps % 1000 == 0 and rank == 0:
                            print("steps", total_steps, " rank", rank)
                        if total_steps % 1000 == 0 and rank == 1:
                            print("steps", total_steps, " rank", rank)
                        for t_train in range(self.nb_train_steps):
                            # Not enough samples in the replay buffer
                            if not self.replay_buffer.can_sample(self.batch_size) or total_steps < 1190:
                                MPI.COMM_WORLD.Barrier()
                                break
                            MPI.COMM_WORLD.Barrier()

                            # Adapt param noise, if necessary.
                            if len(self.replay_buffer) >= self.batch_size and \
                                    t_train % self.param_noise_adaption_interval == 0:
                                distance = self._adapt_param_noise()
                                epoch_adaptive_distances.append(distance)

                            # weird equation to deal with the fact the nb_train_steps will be different
                            # to nb_rollout_steps
                            step = (int(t_train * (self.nb_rollout_steps / self.nb_train_steps)) +
                                    self.num_timesteps - self.nb_rollout_steps)

                            critic_loss, actor_loss = self._train_step(step, writer, log=t_train == 0)
                            epoch_critic_losses.append(critic_loss)
                            epoch_actor_losses.append(actor_loss)
                            self._update_target_net()

                        # Evaluate.
                        eval_episode_rewards = []
                        eval_qs = []
                        MPI.COMM_WORLD.Barrier()
                        if self.eval_env is not None:
                            eval_episode_reward = 0.
                            for _ in range(self.nb_eval_steps):
                                if total_steps >= total_timesteps:
                                    return self

                                eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                                unscaled_action = unscale_action(self.action_space, eval_action)
                                eval_obs, eval_r, eval_done, _ = self.eval_env.step(unscaled_action)
                                if self.render_eval:
                                    self.eval_env.render()
                                eval_episode_reward += eval_r

                                eval_qs.append(eval_q)
                                if eval_done:
                                    if not isinstance(self.env, VecEnv):
                                        eval_obs = self.eval_env.reset()
                                    eval_episode_rewards.append(eval_episode_reward)
                                    eval_episode_rewards_history.append(eval_episode_reward)
                                    eval_episode_reward = 0.

                    mpi_size = MPI.COMM_WORLD.Get_size()
                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                    combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                    combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                    combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                    combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                    combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                    if len(epoch_adaptive_distances) != 0:
                        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rollout/episodes'] = epoch_episodes
                    combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                    # Evaluation statistics.
                    if self.eval_env is not None:
                        combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                        combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                        combined_stats['eval/Q'] = np.mean(eval_qs)
                        combined_stats['eval/episodes'] = len(eval_episode_rewards)

                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/epochs'] = epoch + 1
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.dump_tabular()
                    logger.info('')
                    logdir = logger.get_dir()
                    if rank == 0 and logdir:
                        if hasattr(self.env, 'get_state'):
                            with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.env.get_state(), file_handler)
                        if self.eval_env and hasattr(self.eval_env, 'get_state'):
                            with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.eval_env.get_state(), file_handler)

    def save(self, save_path, cloudpickle=False):
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "nb_eval_steps": self.nb_eval_steps,
            "param_noise_adaption_interval": self.param_noise_adaption_interval,
            "nb_train_steps": self.nb_train_steps,
            "nb_rollout_steps": self.nb_rollout_steps,
            "verbose": self.verbose,
            "param_noise": self.param_noise,
            "action_noise": self.action_noise,
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_returns": self.normalize_returns,
            "enable_popart": self.enable_popart,
            "normalize_observations": self.normalize_observations,
            "batch_size": self.batch_size,
            "observation_range": self.observation_range,
            "return_range": self.return_range,
            "critic_l2_reg": self.critic_l2_reg,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "clip_norm": self.clip_norm,
            "reward_scale": self.reward_scale,
            "memory_limit": self.memory_limit,
            "buffer_size": self.buffer_size,
            "random_exploration": self.random_exploration,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "expert_use": self.expert_use,
            "expert_data": self.expert_data,
            "expert_batch_size": self.expert_batch_size,
            "expert_batch_size_current": self.expert_batch_size_current,
            "expert_limit_success": self.expert_limit_success
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path,
                           data=data,
                           params=params_to_save,
                           cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()
        # Patch for version < v2.6.0, duplicated keys where saved
        if len(params) > len(model.get_parameter_list()):
            n_params = len(model.params)
            n_target_params = len(model.target_params)
            n_normalisation_params = len(model.obs_rms_params) + len(model.ret_rms_params)
            # Check that the issue is the one from
            # https://github.com/hill-a/stable-baselines/issues/363
            assert len(params) == 2 * (n_params + n_target_params) + n_normalisation_params, \
                "The number of parameter saved differs from the number of parameters" \
                " that should be loaded: {}!={}".format(len(params), len(model.get_parameter_list()))
            # Remove duplicates
            params_ = params[:n_params + n_target_params]
            if n_normalisation_params > 0:
                params_ += params[-n_normalisation_params:]
            params = params_
        model.load_parameters(params)

        if model.expert_use:
            model._init_demo_buffer()

        return model
