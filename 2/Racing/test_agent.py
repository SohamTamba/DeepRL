from __future__ import print_function

from datetime import datetime
import numpy as np
import torch
import gym
import os
import json
import argparse

from model import Model
import utils
from dataloader import process_img

import cv2


def run_episode(env, agent, save_states, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    agent.begin_episode()
    device = utils.get_device()

    state_history = []
    if save_states:
        state_history.append(state)
    while True:
        with torch.no_grad():
            state = process_img(state)
            state = state.to(device)
            a = agent.infer_action(state)
            a = a.cpu().numpy()

        next_state, r, done, info = env.step(a)   
        episode_reward += r
        state = next_state
        step += 1

        if save_states:
            state_history.append(next_state)

        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward, state_history


def main(opt):

    rendering = opt.render
    
    n_test_episodes = opt.num_episodes
    device = utils.get_device()
    print(f"device = {device}")

    agent = Model(opt, opt.model_dir).to(device)
    agent.load()
    agent.eval()

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    if opt.save_video:
        vid_path  = f'{opt.output_dir}/{agent.title}.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(vid_path, fourcc, 30, (96, 96))
        print(f"Saving to {vid_path}")
    for i in range(n_test_episodes):
        episode_reward, frames = run_episode(env, agent, opt.save_video, rendering=rendering)
        episode_rewards.append(episode_reward)
        if opt.save_video:
            for f in frames:
                video.write(f)

    video.release()
    cv2.destroyAllWindows()



    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = os.path.join(opt.output_dir, f"{agent.title}.json")
    fh = open(fname, "w")
    json.dump(results, fh)
    print(results)

    env.close()
    print('... finished')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir', type=str, default="./test_output")
    parser.add_argument('-model_dir', type=str, default="./train_output")
    parser.add_argument('-N_history', type=int, default=8)
    parser.add_argument('-num_train', type=int, default=-1, help="Set to negative to use all remaining training data")
    parser.add_argument('-encoder', type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument('--freeze_encoder', action="store_true")
    parser.add_argument('-num_decoder_layers', type=int, default=1)
    parser.add_argument('-num_episodes', type=int, default=15)
    parser.add_argument('-prediction_mode', type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument('--download_encoder', action="store_true")
    parser.add_argument('--save_video', action="store_true")
    parser.add_argument('--render', action="store_true")



    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
