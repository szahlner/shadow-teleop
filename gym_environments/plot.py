import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import fnmatch
import gym
import gym_shadow_hand

ALGOS = ["DDPG", "HER+DDPG", "DDPGfED", "HER+DDPGfED"]
ALGOS.extend([algo.lower() for algo in ALGOS])

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="shadow_hand_reach-v1", help='environment ID')
    parser.add_argument('--algo', help='RL Algorithm', default='DDPGfED', type=str, nargs='+', choices=ALGOS)
    parser.add_argument('--agent', help='Location of the agent', type=str, nargs='+', required=True)
    parser.add_argument('--window-size', help='Window size', type=int, default=5)
    parser.add_argument('--single-worker', help='Use single workers', action='store_true', required=False)
    args = parser.parse_args()

    env = gym.make(args.env)
    
    legacy_env = False
    if args.env.endswith("v0"):
        legacy_env = True

    n_agents = len(args.agent)
    window_size = args.window_size

    if n_agents is not len(args.algo):
        args.algo = [args.algo] * n_agents

    expert_batch_size_all, training_success_median = [], []
    result_median, result_min, result_max = [], [], []
    success_median, success_min, success_max = [], [], []
    x, x_batch = [], []

    eval_episodes = False

    for n in range(n_agents):
        try:
            agent = args.agent[n].split(":")[0]
        except IndexError:
            agent = args.agent[n]

        if "her" in args.algo[n] or "HER" in args.algo[n]:
            args.algo[n] = "her"

        log_dir = os.path.join(os.path.dirname(__file__), "logs", args.algo[n].lower(), agent)
        logs = fnmatch.filter(os.listdir(log_dir), "*.npz")
        n_worker = len(logs)

        # Get length
        path = os.path.join(log_dir, logs[0])
        data = np.load(path)
        n_eval, n_eval_ep, ep_length = data["results"].shape[0], data["results"].shape[1], data["results"].shape[2]

        if "episode" in logs[0]:
            x.append(data["episodes"])
            eval_episodes = True
        else:
            if legacy_env or args.single_worker:
                x.append(data["timesteps"])
            else:
                x.append(data["timesteps"] * n_worker)
        x_batch.append(x[n])

        try:
            t = data["expert_batch_sizes"]
            del t
            has_expert_batch_size = True
        except KeyError:
            has_expert_batch_size = False

        del data

        if has_expert_batch_size:
            expert_batch_size = np.empty(shape=(n_eval, n_worker))

        result = np.empty(shape=(n_eval, n_worker))
        success = np.empty(shape=(n_eval, n_worker))
        training_success = np.empty(shape=(n_eval, n_worker))

        for k in range(n_worker):
            path = os.path.join(log_dir, logs[k])
            data = np.load(path)

            if has_expert_batch_size and not legacy_env:
                expert_batch_size[:, k] = data["expert_batch_sizes"]

            if legacy_env:
                if ep_length is not env.max_steps_per_episode + 1:
                    result_ = np.max(data["results"][:, :, :env.max_steps_per_episode + 1], axis=2)
                    success_ = np.sum(data["successes"][:, :, :env.max_steps_per_episode + 1], axis=2)
                else:
                    result_ = np.max(data["results"], axis=2)
                    success_ = np.sum(data["successes"], axis=2)

                result[:, k] = np.mean(result_, axis=1)
                success_ = np.where(success_ > 0, 1., 0.)
                success[:, k] = np.mean(success_, axis=1)
            else:
                if ep_length is not env.max_steps_per_episode + 1:
                    result_ = np.mean(data["results"][:, :, :env.max_steps_per_episode + 1], axis=2)
                    success_ = np.mean(data["successes"][:, :, :env.max_steps_per_episode + 1], axis=2)
                else:
                    result_ = np.mean(data["results"], axis=2)
                    success_ = np.mean(data["successes"], axis=2)

                result[:, k] = np.mean(result_, axis=1)
                success[:, k] = np.mean(success_, axis=1)

            if n_eval < len(data["training_successes"]):
                step = int(len(data["training_successes"]) / n_eval)
                training_success[:, k] = data["training_successes"][::step]
            elif n_eval > len(data["training_successes"]):
                training_successes = data["training_successes"]
                training_successes = np.concatenate((training_successes, np.array([data["training_successes"][-1]] * (n_eval - len(data["training_successes"])))), axis=0)
                training_success[:, k] = training_successes
            else:
                training_success[:, k] = data["training_successes"]

            del data

        if has_expert_batch_size:
            expert_batch_size = np.median(expert_batch_size, axis=1)
            expert_batch_size_all.append(expert_batch_size)

        result_max_, result_min_ = np.percentile(result, [75, 25], axis=1)
        result = np.median(result, axis=1)

        success_max_, success_min_ = np.percentile(success, [75, 25], axis=1)
        success = np.median(success, axis=1)

        training_success = np.median(training_success, axis=1)

        result_median.append(moving_average(result, window=window_size))
        result_min.append(moving_average(result_min_, window=window_size))
        result_max.append(moving_average(result_max_, window=window_size))

        success_median.append(moving_average(success, window=window_size))
        success_min.append(moving_average(success_min_, window=window_size))
        success_max.append(moving_average(success_max_, window=window_size))

        training_success_median.append(moving_average(training_success, window=window_size))

        x[n] = x[n][len(x[n]) - len(success_median[n]):]

    if not legacy_env:
        expert_batch_size_all = np.array(expert_batch_size_all)

    result_median = np.array(result_median)
    result_min = np.array(result_min)
    result_max = np.array(result_max)

    success_median = np.array(success_median)
    success_min = np.array(success_min)
    success_max = np.array(success_max)

    training_success_median = np.array(training_success_median)

    with plt.style.context("ggplot"):
        color = [(0., 0.4470, 0.7410, 1.), (0.8500, 0.3250, 0.0980, 1.), (0.4660, 0.6740, 0.1880, 1.)]

        if args.env in ["shadow_hand_reach-v1", "shadow_hand_reach_goalenv-v1"]:
            title = "ShadowHandReach-v1"
        elif args.env in ["shadow_hand_block-v1", "shadow_hand_block_goalenv-v1"]:
            title = "ShadowHandBlock-v1"
        elif args.env in ["shadow_hand_reach-v0", "shadow_hand_reach_goalenv-v0"]:
            title = "ShadowHandReach-v0"

        if eval_episodes:
            x_label = "Episodes"
        else:
            x_label = "Timesteps"

        y_label = "Median Success Rate"
        fig, axes = plt.subplots(nrows=1, ncols=1)

        legend = []
        for n in range(n_agents):
            axes.plot(x[n], success_median[n], color=color[n])
            axes.fill_between(x[n], success_min[n], success_max[n], color=color[n], alpha=0.2)

            try:
                legend.append(args.agent[n].split(":")[1])
            except IndexError:
                legend.append(args.algo[n])

        axes.set_ylim([-0.05, 1.05])
        axes.set_xlabel(x_label, color="k")
        axes.set_ylabel(y_label, color="k")
        if eval_episodes:
            axes.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
            #axes.set_xlim(0, 15000)
        else:
            axes.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))
        axes.tick_params(axis="both", colors="k")
        axes.legend(legend, loc="lower right")
        axes.set_title(title, color="k")

        y_label = "Median Training Success Rate"
        fig, axes = plt.subplots(nrows=1, ncols=1)

        for n in range(n_agents):
            axes.plot(x[n], training_success_median[n], color=color[n])

        axes.set_xlabel(x_label, color="k")
        axes.set_ylabel(y_label, color="k")
        if eval_episodes:
            axes.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
        else:
            axes.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))
        axes.tick_params(axis="both", colors="k")
        axes.legend(legend, loc="lower right")
        axes.set_title(title, color="k")

        y_label = "Median Reward"
        fig, axes = plt.subplots(nrows=1, ncols=1)

        for n in range(n_agents):
            axes.plot(x[n], result_median[n,], color=color[n])
            axes.fill_between(x[n], result_min[n], result_max[n], color=color[n], alpha=0.2)

        axes.set_xlabel(x_label, color="k")
        axes.set_ylabel(y_label, color="k")
        if eval_episodes:
            axes.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
        else:
            axes.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))
        axes.tick_params(axis="both", colors="k")
        axes.legend(legend, loc="lower right")
        axes.set_title(title, color="k")

        if has_expert_batch_size and not legacy_env:
            y_label = "Expert Batch Size"
            fig, axes = plt.subplots(nrows=1, ncols=1)

            for n in range(n_agents):
                axes.plot(x_batch[n], expert_batch_size_all[n], color=color[n])

            axes.set_xlabel(x_label, color="k")
            axes.set_ylabel(y_label, color="k")
            if eval_episodes:
                axes.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))
            else:
                axes.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))
            axes.tick_params(axis="both", colors="k")
            axes.legend(legend, loc="upper right")
            axes.set_title(title, color="k")

        plt.show()
