import os
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 20
N_EVAL_EPISODES = 5
MAX_EP_LENGTH = 1000
THRESHOLD = -0.01


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


# Without pretrain
# Load results from evaluations
log_dir_0 = "shadow_hand_reach-v0_1_seed_19"
log_dir_0 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_0, "evaluations.npz")
log_dir_1 = "shadow_hand_reach-v0_1_seed_20"
log_dir_1 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_1, "evaluations.npz")
log_dir_2 = "shadow_hand_reach-v0_1_seed_21"
log_dir_2 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_2, "evaluations.npz")
log_dir_3 = "shadow_hand_reach-v0_1_seed_22"
log_dir_3 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_3, "evaluations.npz")
log_dir_4 = "shadow_hand_reach-v0_1_seed_23"
log_dir_4 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_4, "evaluations.npz")

data_0 = np.load(log_dir_0)
data_1 = np.load(log_dir_1)
data_2 = np.load(log_dir_2)
data_3 = np.load(log_dir_3)
data_4 = np.load(log_dir_4)

# x for diagram
x = data_0["timesteps"]

# results - mean per seed and evaluation
results_0 = data_0["results"].reshape(data_0["results"].shape[0], N_EVAL_EPISODES)
results_1 = data_1["results"].reshape(data_1["results"].shape[0], N_EVAL_EPISODES)
results_2 = data_2["results"].reshape(data_2["results"].shape[0], N_EVAL_EPISODES)
results_3 = data_3["results"].reshape(data_3["results"].shape[0], N_EVAL_EPISODES)
results_4 = data_4["results"].reshape(data_4["results"].shape[0], N_EVAL_EPISODES)

# episode lengths - mean per seed and evaluation
ep_lengths_0 = data_0["ep_lengths"]
ep_lengths_1 = data_1["ep_lengths"]
ep_lengths_2 = data_2["ep_lengths"]
ep_lengths_3 = data_3["ep_lengths"]
ep_lengths_4 = data_4["ep_lengths"]

# success - less than max steps and results < threshold
success_0 = (ep_lengths_0 < MAX_EP_LENGTH).astype(np.float32)
success_1 = (ep_lengths_1 < MAX_EP_LENGTH).astype(np.float32)
success_2 = (ep_lengths_2 < MAX_EP_LENGTH).astype(np.float32)
success_3 = (ep_lengths_3 < MAX_EP_LENGTH).astype(np.float32)
success_4 = (ep_lengths_4 < MAX_EP_LENGTH).astype(np.float32)

results_0 = np.mean(results_0, axis=1)
results_1 = np.mean(results_1, axis=1)
results_2 = np.mean(results_2, axis=1)
results_3 = np.mean(results_3, axis=1)
results_4 = np.mean(results_4, axis=1)

success_0 = np.mean(success_0, axis=1)
success_1 = np.mean(success_1, axis=1)
success_2 = np.mean(success_2, axis=1)
success_3 = np.mean(success_3, axis=1)
success_4 = np.mean(success_4, axis=1)

# y for diagram
y = np.array([results_0, results_1, results_2, results_3, results_4])
y_success = np.array([success_0, success_1, success_2, success_3, success_4])

y_max, y_min = np.percentile(y, [75, 25], axis=0)
y_median = np.median(y, axis=0)

y_min = moving_average(y_min, window=WINDOW_SIZE)
y_max = moving_average(y_max, window=WINDOW_SIZE)
y_median = moving_average(y_median, window=WINDOW_SIZE)
x = x[len(x) - len(y_median):]

y_max_success, y_min_success = np.percentile(y_success, [75, 25], axis=0)
y_median_success = np.median(y_success, axis=0)

y_min_success = moving_average(y_min_success, window=WINDOW_SIZE)
y_max_success = moving_average(y_max_success, window=WINDOW_SIZE)
y_median_success = moving_average(y_median_success, window=WINDOW_SIZE)

"""
# Without pretrain
# Load results from evaluations
log_dir_0 = "shadow_hand_reach_goalenv-v0_1_seed_19"
log_dir_0 = os.path.join(os.path.dirname(__file__), "logs", "her", log_dir_0, "evaluations.npz")
log_dir_1 = "shadow_hand_reach_goalenv-v0_1_seed_20"
log_dir_1 = os.path.join(os.path.dirname(__file__), "logs", "her", log_dir_1, "evaluations.npz")
log_dir_2 = "shadow_hand_reach_goalenv-v0_1_seed_21"
log_dir_2 = os.path.join(os.path.dirname(__file__), "logs", "her", log_dir_2, "evaluations.npz")
#log_dir_3 = "shadow_hand_reach_goalenv-v0_1_seed_22"
#log_dir_3 = os.path.join(os.path.dirname(__file__), "logs", "her", log_dir_3, "evaluations.npz")
#log_dir_4 = "shadow_hand_reach_goalenv-v0_1_seed_23"
#log_dir_4 = os.path.join(os.path.dirname(__file__), "logs", "her", log_dir_4, "evaluations.npz")

data_0 = np.load(log_dir_0)
data_1 = np.load(log_dir_1)
data_2 = np.load(log_dir_2)
#data_3 = np.load(log_dir_3)
#data_4 = np.load(log_dir_4)

# x for diagram
x_her = data_0["timesteps"]

# results - mean per seed and evaluation
results_0 = np.mean(data_0["results"].reshape(data_0["results"].shape[0], N_EVAL_EPISODES), axis=1)
results_1 = np.mean(data_1["results"].reshape(data_1["results"].shape[0], N_EVAL_EPISODES), axis=1)
results_2 = np.mean(data_2["results"].reshape(data_2["results"].shape[0], N_EVAL_EPISODES), axis=1)
#results_3 = np.mean(data_3["results"].reshape(data_3["results"].shape[0], N_EVAL_EPISODES), axis=1)
#results_4 = np.mean(data_4["results"].reshape(data_4["results"].shape[0], N_EVAL_EPISODES), axis=1)

# success
success_0 = np.mean(data_0["successes"], axis=1)
success_1 = np.mean(data_1["successes"], axis=1)
success_2 = np.mean(data_2["successes"], axis=1)
#success_3 = np.mean(data_3["successes"], axis=1)
#success_4 = np.mean(data_4["successes"], axis=1)

# y for diagram
y = np.array([results_0, results_1, results_2])
y_success = np.array([success_0, success_1])

y_max_her, y_min_her = np.percentile(y, [75, 25], axis=0)
y_median_her = np.median(y, axis=0)

y_min_her = moving_average(y_min_her, window=WINDOW_SIZE)
y_max_her = moving_average(y_max_her, window=WINDOW_SIZE)
y_median_her = moving_average(y_median_her, window=WINDOW_SIZE)
x_her = x_her[len(x_her) - len(y_median_her):]

y_max_success_her, y_min_success_her = np.percentile(y_success, [75, 25], axis=0)
y_median_success_her = np.median(y_success, axis=0)

y_min_success_her = moving_average(y_min_success_her, window=WINDOW_SIZE)
y_max_success_her = moving_average(y_max_success_her, window=WINDOW_SIZE)
y_median_success_her = moving_average(y_median_success_her, window=WINDOW_SIZE)
"""

# Pretrain 1000 iterations
log_dir_1000_0 = "shadow_hand_reach-v0_1_seed_19_pretrain_1000"
log_dir_1000_0 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_1000_0, "evaluations.npz")
log_dir_1000_1 = "shadow_hand_reach-v0_1_seed_20_pretrain_1000"
log_dir_1000_1 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_1000_1, "evaluations.npz")
log_dir_1000_2 = "shadow_hand_reach-v0_1_seed_21_pretrain_1000"
log_dir_1000_2 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_1000_2, "evaluations.npz")
log_dir_1000_3 = "shadow_hand_reach-v0_1_seed_22_pretrain_1000"
log_dir_1000_3 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_1000_3, "evaluations.npz")
log_dir_1000_4 = "shadow_hand_reach-v0_1_seed_23_pretrain_1000"
log_dir_1000_4 = os.path.join(os.path.dirname(__file__), "logs", "ddpg", log_dir_1000_4, "evaluations.npz")

data_1000_0 = np.load(log_dir_1000_0)
data_1000_1 = np.load(log_dir_1000_1)
data_1000_2 = np.load(log_dir_1000_2)
data_1000_3 = np.load(log_dir_1000_3)
data_1000_4 = np.load(log_dir_1000_4)

# x for diagram
x_1000 = data_1000_0["timesteps"]

# results - mean per seed and evaluation
results_1000_0 = data_1000_0["results"].reshape(data_1000_0["results"].shape[0], N_EVAL_EPISODES)
results_1000_1 = data_1000_1["results"].reshape(data_1000_1["results"].shape[0], N_EVAL_EPISODES)
results_1000_2 = data_1000_2["results"].reshape(data_1000_2["results"].shape[0], N_EVAL_EPISODES)
results_1000_3 = data_1000_3["results"].reshape(data_1000_3["results"].shape[0], N_EVAL_EPISODES)
results_1000_4 = data_1000_4["results"].reshape(data_1000_4["results"].shape[0], N_EVAL_EPISODES)

# episode lengths - mean per seed and evaluation
ep_lengths_1000_0 = data_1000_0["ep_lengths"]
ep_lengths_1000_1 = data_1000_1["ep_lengths"]
ep_lengths_1000_2 = data_1000_2["ep_lengths"]
ep_lengths_1000_3 = data_1000_3["ep_lengths"]
ep_lengths_1000_4 = data_1000_4["ep_lengths"]

# success - less than max steps and results < threshold
success_1000_0 = (ep_lengths_1000_0 < MAX_EP_LENGTH).astype(np.float32)
success_1000_1 = (ep_lengths_1000_1 < MAX_EP_LENGTH).astype(np.float32)
success_1000_2 = (ep_lengths_1000_2 < MAX_EP_LENGTH).astype(np.float32)
success_1000_3 = (ep_lengths_1000_3 < MAX_EP_LENGTH).astype(np.float32)
success_1000_4 = (ep_lengths_1000_4 < MAX_EP_LENGTH).astype(np.float32)

results_1000_0 = np.mean(results_1000_0, axis=1)
results_1000_1 = np.mean(results_1000_1, axis=1)
results_1000_2 = np.mean(results_1000_2, axis=1)
results_1000_3 = np.mean(results_1000_3, axis=1)
results_1000_4 = np.mean(results_1000_4, axis=1)

success_1000_0 = np.mean(success_1000_0, axis=1)
success_1000_1 = np.mean(success_1000_1, axis=1)
success_1000_2 = np.mean(success_1000_2, axis=1)
success_1000_3 = np.mean(success_1000_3, axis=1)
success_1000_4 = np.mean(success_1000_4, axis=1)

# y for diagram
y_1000 = np.array([results_1000_0, results_1000_2, results_1000_3, results_1000_4])
y_1000_success = np.array([success_1000_0, success_1000_2, success_1000_3, success_1000_4])

y_1000_max, y_1000_min = np.percentile(y_1000, [75, 25], axis=0)
y_1000_median = np.median(y_1000, axis=0)

y_1000_min = moving_average(y_1000_min, window=WINDOW_SIZE)
y_1000_max = moving_average(y_1000_max, window=WINDOW_SIZE)
y_1000_median = moving_average(y_1000_median, window=WINDOW_SIZE)
x_1000 = x_1000[len(x_1000) - len(y_1000_median):]

y_1000_max_success, y_1000_min_success = np.percentile(y_1000_success, [75, 25], axis=0)
y_1000_median_success = np.median(y_1000_success, axis=0)

y_1000_min_success = moving_average(y_1000_min_success, window=WINDOW_SIZE)
y_1000_max_success = moving_average(y_1000_max_success, window=WINDOW_SIZE)
y_1000_median_success = moving_average(y_1000_median_success, window=WINDOW_SIZE)

with plt.style.context("ggplot"):
    color = (0., 0.4470, 0.7410, 1.)
    color_10 = (0.4660, 0.6740, 0.1880, 1.)
    color_100 = (0.4940, 0.1840, 0.5560, 1.)
    color_1000 = (0.8500, 0.3250, 0.0980, 1.)
    color_10000 = (0.6350, 0.0780, 0.1840, 1.)
    color_100000 = (0.9290, 0.6940, 0.1250, 1.)

    title = "ShadowHandReach-v0"
    x_label = "Timesteps"
    y_label = "Median Reward"

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(x, y_median, color=color)
    axes.fill_between(x, y_min, y_max, color=color, alpha=0.2)
    axes.set_ylim([None, 5])
    axes.set_xlabel(x_label, color="k")
    axes.set_ylabel(y_label, color="k")
    axes.tick_params(axis="both", colors="k")
    axes.legend(["DDPG dense"], loc="lower right")
    axes.set_title(title, color="k")

    fig.savefig(os.path.join("images",
                             "shadow_hand_reach",
                             "ddpg",
                             "shadow_hand_reach-v0.png"), format="png")
    fig.savefig(os.path.join("images",
                             "shadow_hand_reach",
                             "ddpg",
                             "shadow_hand_reach-v0.svg"), format="svg")
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(x, y_median, color=color)
    axes.fill_between(x, y_min, y_max, color=color, alpha=0.2)
    axes.plot(x_1000, y_1000_median, color=color_1000)
    axes.fill_between(x_1000, y_1000_min, y_1000_max, color=color_1000, alpha=0.2)
    # axes.plot(x_her, y_median_her, color=color_10)
    # axes.fill_between(x_her, y_min_her, y_max_her, color=color_10, alpha=0.2)
    axes.set_ylim([None, 5])
    axes.set_xlabel(x_label, color="k")
    axes.set_ylabel(y_label, color="k")
    axes.tick_params(axis="both", colors="k")
    axes.legend(["DDPG dense", "DDPG dense w/p 1000"], loc="lower right")
    axes.set_title(title, color="k")

    fig.savefig(os.path.join("images",
                             "shadow_hand_reach",
                             "ddpg",
                             "shadow_hand_reach-v0_pretrain_1000.png"), format="png")
    fig.savefig(os.path.join("images",
                             "shadow_hand_reach",
                             "ddpg",
                             "shadow_hand_reach-v0_pretrain_1000.svg"), format="svg")
    plt.close()

    y_label = "Median Success Rate"

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(x, y_median_success, color=color)
    axes.fill_between(x, y_min_success, y_max_success, color=color, alpha=0.2)
    axes.set_ylim([-0.05, 1.05])
    axes.set_xlabel(x_label, color="k")
    axes.set_ylabel(y_label, color="k")
    axes.tick_params(axis="both", colors="k")
    axes.legend(["DDPG dense"], loc="upper left")
    axes.set_title(title, color="k")

    fig.savefig(os.path.join("images",
                             "shadow_hand_reach",
                             "ddpg",
                             "shadow_hand_reach-v0_success.png"), format="png")
    fig.savefig(os.path.join("images",
                             "shadow_hand_reach",
                             "ddpg",
                             "shadow_hand_reach-v0_success.svg"), format="svg")
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=1) # , figsize=(6, 4))
    axes.plot(x, y_median_success, color=color)
    axes.fill_between(x, y_min_success, y_max_success, color=color, alpha=0.2)
    axes.plot(x_1000, y_1000_median_success, color=color_1000)
    axes.fill_between(x_1000, y_1000_min_success, y_1000_max_success, color=color_1000, alpha=0.2)
    # axes.plot(x_her, y_median_success_her, color=color_10)
    # axes.fill_between(x_her, y_min_success_her, y_max_success_her, color=color_10, alpha=0.2)
    axes.set_ylim([-0.05, 1.05])
    axes.set_xlabel(x_label, color="k")
    axes.set_ylabel(y_label, color="k")
    axes.tick_params(axis="both", colors="k")
    axes.legend(["DDPG dense", "DDPG dense w/p 1000"], loc="upper left")
    axes.set_title(title, color="k")

    fig.savefig(os.path.join("images",
                             "shadow_hand_reach",
                             "ddpg",
                             "shadow_hand_reach-v0_pretrain_1000_success.png"), format="png")
    fig.savefig(os.path.join("images",
                             "shadow_hand_reach",
                             "ddpg",
                             "shadow_hand_reach-v0_pretrain_1000_success.svg"), format="svg")
    plt.close()
