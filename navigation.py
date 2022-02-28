import torch
from DQN_agent import DQNAgent
from collections import deque
from navigationEnv import NavigationEnv
import numpy as np
import matplotlib.pyplot as plt


def mov_avg(data, window):
    v = deque([] * window)

    ma_data = []

    for d in data:
        v.append(d)
        ma_data.append(np.average(v))

    return ma_data


def main(train=True):
    # define the device to run the code into: GPU when available, CPU otherwise

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    file_path = "Banana_Linux/Banana.x86"
    env = NavigationEnv(file_path)

    # Network parameters
    gamma = 0.99
    eps = 1.0
    eps_min = 5e-2
    eps_decay = 0.98

    # Final score
    final_score = 13.0

    # Create agent
    agent = DQNAgent(gamma, env.state_size, env.action_size, device, eps=eps, eps_min=eps_min, eps_decay=eps_decay)

    best_weight_path = "navigation_weight.pt"

    if not train:
        agent.load(best_weight_path)

    # Set number of episodes
    n_episodes = 2000
    # Set timeout
    max_t = 500

    # list containing scores from each episode
    scores = []
    # last 100 scores
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):

        state, score = env.reset()

        for t in range(max_t):
            action = agent.egreedy_policy(state)
            next_state, reward, done, _ = env.step(action)

            if train:
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
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        # Check if we hit the final score
        if np.mean(scores_window) >= final_score and train:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            agent.save(best_weight_path)
            # break

    # Average all scores
    window_avg = 100
    ma_data = mov_avg(scores, window_avg)

    plt.plot(scores, alpha=0.5)
    plt.plot(ma_data, alpha=1)
    plt.ylabel('Rewards')
    plt.show()


if __name__ == "__main__":
    # Set train to True to train

    # Set train to False to run the best weight

    main(train=True)





