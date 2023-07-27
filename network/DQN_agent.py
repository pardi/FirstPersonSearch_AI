import torch
import torch.nn.functional as F
import numpy as np
from network import QNetwork
from ReplayBuffer import ReplayBuffer
import random


class DQNAgent:
    def __init__(self, params: dict):
        self.gamma = params["gamma"]
        self.eps = params["eps"]
        self.eps_min = params["eps_min"]
        self.eps_decay = params["eps_decay"]
        self.action_size = params["action_size"]
        self.device = params["device"]
        self.qnetwork_target = QNetwork(params["state_size"], self.action_size).to(self.device)
        self.qnetwork_train = QNetwork(params["state_size"], self.action_size).to(self.device)
        self.optimiser = torch.optim.Adam(self.qnetwork_train.parameters(), lr=params["opt_lr"])
        self.tau = params["tau"]
        self.memory = ReplayBuffer(params["replay_buffer_size"], params["batch_size"], self.device)
        self.t_step = 0
        self.every_update = params["target_update"]

    def egreedy_policy(self, state: np.ndarray) -> np.ndarray:

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Turn Off the training mode
        self.qnetwork_train.eval()

        with torch.no_grad():
            action_values = self.qnetwork_train(state)

        # Turn On the training mode
        self.qnetwork_train.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experience) -> None:

        # Unpack the experience tuple
        states, actions, rewards, next_states, dones = experience

        qnn_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        qnn_targets = rewards + self.gamma * qnn_targets_next * (1 - dones)

        # Expected Qs for each actions
        q_expected = self.qnetwork_train(states).gather(1, actions)

        loss = F.mse_loss(q_expected, qnn_targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.soft_update()

    def soft_update(self) -> None:
        # Soft update model parameters.
        # theta_target = tau * theta_local + (1 - tau) * theta_target

        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_train.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, done: np.ndarray) -> None:

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.every_update
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def update_eps(self) -> None:
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

    def save(self, weight_path) -> None:
        torch.save(self.qnetwork_train.state_dict(), weight_path)

    def load(self, weight_path) -> None:
        self.qnetwork_train.load_state_dict(torch.load(weight_path))
        self.qnetwork_target.load_state_dict(torch.load(weight_path))
