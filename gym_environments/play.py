import os
import numpy as np
import imageio
import cv2
import warnings
import argparse
import pickle

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
from stable_baselines import HER
from models import DDPGfED
from models import DDPGfMPI as DDPG
import gym_shadow_hand


ALGOS = {
    "ddpg": DDPG,
    "her": HER,
    "ddpgfed": DDPGfED
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="shadow_hand_reach-v1", help='environment ID')
    parser.add_argument('--algo', help='RL Algorithm', default='ddpgfed',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--play-episodes', help='Number of episodes to play', default=10, type=int)
    parser.add_argument('--agent', help='Location of the agent', type=str, required=True)
    args = parser.parse_args()

    env_id = args.env
    env = gym.make(env_id, reward_type='dense')
    episode_length = env.max_steps_per_episode

    font = cv2.QT_FONT_NORMAL
    font_scale = 1
    line_type = 2
    if env_id in ["shadow_hand_reach-v0", "shadow_hand_reach_goalenv-v0",
                  "shadow_hand_reach-v1", "shadow_hand_reach_goalenv-v1"]:
        bottom_left_corner_of_text = (175, 440)
        font_color = (0, 0, 0)
    elif env_id in ["shadow_hand_block-v1", "shadow_hand_block_goalenv-v1"]:
        bottom_left_corner_of_text = (175, 475)
        font_color = (0, 0, 0)

    model_path = os.path.join(os.path.dirname(__file__), "logs", args.algo, args.agent, "{}.zip".format(env_id))
    model = ALGOS[args.algo].load(model_path, env=env)

    # Trained agent
    print("Trained agent")
    images = []
    demo = []
    obs = env.reset()
    episode = 1
    for n in range(episode_length * args.play_episodes):
        action, _ = model.predict(obs)
        obs_next, reward, done, info = env.step(action)
        demo.append((obs, action, reward, obs_next, bool(info["is_success"])))
        obs = obs_next
#        img = env.render(mode="rgb_array")
#        cv2.putText(img, "Episode {}".format(episode), fontFace=font, fontScale=font_scale, color=font_color,
#                    lineType=line_type, org=bottom_left_corner_of_text)
#        img = img[:, :, :-1]
#        img[np.where((img == [255, 255, 255]).all(axis=2))] = [101, 158, 199]
#        images.append(img)
        if done:
            obs = env.reset()
            episode += 1

            if episode > args.play_episodes:
                break

    with open("demo.pkl", "wb") as f:
        pickle.dump(demo, f)

#    imageio.mimsave("{}_{}.gif".format(args.algo, env_id),
#                    [np.array(img) for n, img in enumerate(images) if n % 3 == 0],
#                    fps=10)
    print("Trained agent done")
    exit()
    # Random agent
    print("Random agent")
    images = []
    obs = env.reset()
    episode = 1
    for n in range(episode_length * args.play_episodes):
        try:
            action = model.get_random_action()
        except:
            action = np.random.uniform(-1.5, 1.5, model.env.action_space.shape[0])

        obs, reward, done, _ = env.step(action)
        img = env.render(mode="rgb_array")
        cv2.putText(img, "Episode {}".format(episode), fontFace=font, fontScale=font_scale, color=font_color,
                    lineType=line_type, org=bottom_left_corner_of_text)
        img = img[:, :, :-1]
        img[np.where((img == [255, 255, 255]).all(axis=2))] = [101, 158, 199]
        images.append(img)
        if done:
            obs = env.reset()
            episode += 1

            if episode > args.play_episodes:
                break

    imageio.mimsave("{}_{}_random.gif".format(args.algo, env_id),
                    [np.array(img) for n, img in enumerate(images) if n % 3 == 0],
                    fps=10)
    print("Random agent done")