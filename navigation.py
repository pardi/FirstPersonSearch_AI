import torch
from network.network import DQNAgent
from collections import deque
from navigationEnv import NavigationEnv
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

import logging
logging.basicConfig(level=logging.INFO)


def mov_avg(data: list, window: int) -> list:
    values = deque([] * window)

    ma_data = []

    for d in data:
        values.append(d)
        ma_data.append(np.average(values))

    return ma_data


def run_network(params: dict) -> None:

    # define the device to run the code into: GPU when available, CPU otherwise
    device = torch.device(params["use_gpu"] if torch.cuda.is_available() else "cpu")

    # Target/Final score
    final_score = 13.0

    env = NavigationEnv(params["file_env_path"])

    # Network parameters
    agent_params = {"gamma": params["gamma"],
                    "state_size": env.state_size,
                    "action_size": env.action_size,
                    "device": device,
                    "eps": 1.0,
                    "eps_min": 5e-2,
                    "eps_decay": 0.98,
                    "replay_buffer_size": int(1e6),
                    "batch_size": 128}

    # Create agent
    agent = DQNAgent(agent_params)

    best_weight_path = os.path.join(params["best_weight_folder"], "navigation_weight.pt")

    if not params["train"]:
        agent.load(best_weight_path)

    # list containing scores from each episode
    scores = []
    # last 100 scores
    scores_window = deque(maxlen=100)

    for i_episode in range(1, params["num_episodes"] + 1):

        state, score = env.reset()

        for t in range(params["timeout"]):
            action = agent.egreedy_policy(state)
            next_state, reward, done, _ = env.step(action)

            if params["train"]:
                agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        # save most recent score
        scores_window.append(score)
        # save most recent score
        scores.append(score)

        agent.update_eps()

        if i_episode % 100 == 0:
            print('\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window)}')

        # Check if we hit the final score
        if params["train"] and np.mean(scores_window) >= final_score:
            print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window)}")
            agent.save(best_weight_path)

    # Average all scores
    window_avg = 100
    ma_data = mov_avg(scores, window_avg)

    plt.plot(scores, alpha=0.5)
    plt.plot(ma_data, alpha=1)
    plt.ylabel('Rewards')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This program trains and/or executes a DQN agent to solve the '
                                                 'Banana search environment.')

    parser.add_argument('--train', action='store_true', help='Run a single episode', choices=[True, False],
                        default=False)
    parser.add_argument('--num_episodes', action='store_true', help='Number of episodes to run in training mode',
                        default=1)
    parser.add_argument('--best_weight_folder', action='store_true', help='Folder storing the weights for the network',
                        default="best_weights/")
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU',  choices=[True, False], default=True)
    parser.add_argument('--timeout', action='store_true', help='Set timeout', default=500)
    args = parser.parse_args()

    param = {"train": args.train,
             "num_episodes": args.num_episodes,
             "best_weight_folder": args.best_weight_folder,
             "use_gpu": "cuda:0" if args.use_gpu == "True" else "cpu",
             "gamma": 0.99,
             "file_env_path": "Banana_Linux/Banana.x86",
             "timeout": args.timeout}

    run_network(param)



