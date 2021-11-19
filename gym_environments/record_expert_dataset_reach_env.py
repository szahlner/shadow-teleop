import os
import numpy as np
import imageio
import argparse
import warnings

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorboard')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import gym_shadow_hand
from gym_shadow_hand.lib.leap_motion import Leap
from gym_shadow_hand.lib import utils as leap_motion_utils

from stable_baselines.gail import generate_expert_traj
import pybullet
from gym_shadow_hand.envs.reach_env import MOVABLE_JOINTS

EXPERT_DIR = "experts"
os.makedirs(EXPERT_DIR, exist_ok=True)


def get_action():
    action = [0.] * env.action_space.shape[0]

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
                theta1, theta2, theta3, theta4, theta5 = leap_motion_utils.thumb_joint_rotations(hand)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3) and not np.isnan(
                        theta4) and not np.isnan(theta5):
                    action[15:] = [
                        np.clip(theta1, -1.047, 1.047),
                        np.clip(theta2, 0., 1.222),
                        np.clip(theta3, -0.209, 0.209),
                        np.clip(theta4, -0.524, 0.524),
                        np.clip(theta5, 0., 1.571)]

                # INDEX finger
                index_finger = hand.fingers.finger_type(Leap.Finger.TYPE_INDEX)[0]

                theta1, theta2, theta3, _ = leap_motion_utils.finger_joint_rotations(index_finger)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
                    action[2:5] = [
                        np.clip(theta1, -0.349, 0.349),
                        np.clip(theta2, 0., 1.571),
                        np.clip(theta3, 0., 1.571)]

                # MIDDLE finger
                middle_finger = hand.fingers.finger_type(Leap.Finger.TYPE_MIDDLE)[0]

                theta1, theta2, theta3, _ = leap_motion_utils.finger_joint_rotations(middle_finger)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
                    action[5:8] = [
                        np.clip(theta1, -0.349, 0.349),
                        np.clip(theta2, 0., 1.571),
                        np.clip(theta3, 0., 1.571)]

                # RING finger
                ring_finger = hand.fingers.finger_type(Leap.Finger.TYPE_RING)[0]

                theta1, theta2, theta3, _ = leap_motion_utils.finger_joint_rotations(ring_finger)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
                    action[8:11] = [
                        -np.clip(theta1, -0.349, 0.349),  # "-" = BUG ?
                        np.clip(theta2, 0., 1.571),
                        np.clip(theta3, 0., 1.571)]

                # LITTLE finger
                little_finger = hand.fingers.finger_type(Leap.Finger.TYPE_PINKY)[0]

                theta1, theta2, theta3, _ = leap_motion_utils.finger_joint_rotations(little_finger)

                if not np.isnan(theta1) and not np.isnan(theta2) and not np.isnan(theta3):
                    action[11:15] = [
                        0.,  # Little Finger 5
                        -np.clip(theta1, -0.349, 0.349),  # "-" = BUG ?
                        np.clip(theta2, 0., 1.571),
                        np.clip(theta3, 0., 1.571)]

                # WRIST
                action[0:2] = [
                    -leap_motion_utils.joint_rotation_y(hand.arm.basis, hand.basis, -0.489, 0.140),
                    leap_motion_utils.joint_rotation_x(hand.arm.basis, hand.basis, -0.698, 0.489)]

                # Mark right hand as processed
                right_hand_processed = True

    action = np.array(action).copy()

    if action.shape[0] != env.action_space.shape[0]:
        action = [0.] * env.action_space.shape[0]
        return np.array(action).copy()

    return action


def normalize_actions(action):
    joint_limit_low = []
    joint_limit_high = []
    joints_movable = []

    for n in range(env.n_model_joints):
        joint_info = pybullet.getJointInfo(env.model_id, n)

        if joint_info[1] in MOVABLE_JOINTS:
            joint_limit_low.append(joint_info[8])
            joint_limit_high.append(joint_info[9])
            joints_movable.append(n)

    joint_limit_low = np.array(joint_limit_low)
    joint_limit_high = np.array(joint_limit_high)

    act_range = (joint_limit_high - joint_limit_low) / 2.
    act_center = (joint_limit_high + joint_limit_low) / 2.

    return (action - act_center) / act_range


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="shadow_hand_reach-v0", help="environment ID")
    parser.add_argument("--episodes", help="number of episodes to record", default=10, type=int)
    parser.add_argument("--sub-steps", help="number of sub-steps", default=1, type=int)
    parser.add_argument("--threshold", help="threshold for episode end", default=0.01, type=float)
    parser.add_argument("--render", help="render the record", action="store_true")
    args = parser.parse_args()

    # Expert path
    env_dir = os.path.join(EXPERT_DIR, args.env)
    os.makedirs(env_dir, exist_ok=True)

    save_path = os.path.join(EXPERT_DIR, args.env, "{}_expert".format(args.env))

    # Make environment
    # if args.render:
    #    env = gym.make(args.env, render=True, sim_n_sub_steps=args.sub_steps, distance_threshold=args.threshold)
    # else:
    #    env = gym.make(args.env, render=True, sim_n_sub_steps=args.sub_steps, distance_threshold=args.threshold)

    env = gym.make(args.env, render=True, reward_type="dense")

    # Reset
    env.reset()

    # Leap motion controller
    # Construct after the environment, otherwise PyBullet is not instantiated
    controller = Leap.Controller()

    if args.render:
        images = []

        print("Start rendering...")

        for n in range(args.episodes):
            while True:
                action = get_action()
                action = normalize_actions(action)

                _, _, done, _ = env.step(action)

                img = env.render(mode="rgb_array")
                img = img[:, :, :-1]
                img[np.where((img == [255, 255, 255]).all(axis=2))] = [101, 158, 199]
                images.append(img)

                # Reset time steps
                env.current_episode_steps = 0

                if done:
                    env.reset()
                    break

            print("Recorded {} episodes.".format(n + 1))

        imageio.mimsave("ddpg_shadow_hand_reach_pretraining.gif",
                        [np.array(img) for n, img in enumerate(images) if n % 2 == 0], fps=10)
        env.close()
    else:
        actions = []
        observations = []
        rewards = []
        episode_returns = np.zeros((args.episodes,))
        episode_starts = []

        observation = env.reset()

        episode_starts.append(True)
        reward_sum = 0.

        print("Start recording...")

        for n in range(args.episodes):
            while True:
                observations.append(observation)

                action = get_action()
                action = normalize_actions(action)

                observation, reward, done, info = env.step(action)

                actions.append(action)
                rewards.append(reward)
                episode_starts.append(done)

                reward_sum += reward

                print(reward)

                # Reset time steps
                env.current_episode_steps = 0

                if done or info["is_success"]:
                    episode_returns[n] = reward_sum
                    reward_sum = 0.
                    observation = env.reset()
                    break
            print("Recorded {} episodes.".format(n + 1))

        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
        rewards = np.array(rewards)
        episode_starts = np.array(episode_starts[:-1])

        numpy_dict = {
            "actions": actions,
            "obs": observations,
            "rewards": rewards,
            "episode_returns": episode_returns,
            "episode_starts": episode_starts
        }

        for key, val in numpy_dict.items():
            print(key, val.shape)

        np.savez_compressed(save_path, **numpy_dict)

        env.close()
