import torch
import torch.nn.functional as F
import numpy as np
from network import QNetwork
from ReplayBuffer import ReplayBuffer
import random


class DQNAgent(object):
    def __init__(self, gamma, state_size, action_size, device, eps=1.0, eps_min=0.006, eps_decay=0.998, opt_lr=1e-4, replay_buffer_size=int(1e5), batch_size=64, tau=1e-3, target_update=10):
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.action_size = action_size
        self.device = device
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.qnetwork_train = QNetwork(state_size, action_size).to(device)
        self.optimiser = torch.optim.Adam(self.qnetwork_train.parameters(), lr=opt_lr)
        self.tau = tau
        self.memory = ReplayBuffer(replay_buffer_size, batch_size, device)
        self.t_step = 0
        self.every_update = target_update

    def egreedy_policy(self, state):

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

    def learn(self, experience):

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

    def soft_update(self):
        # Soft update model parameters.
        # theta_target = tau * theta_local + (1 - tau) * theta_target

        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_train.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.every_update
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def update_eps(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

    def save(self, weight_path):
        torch.save(self.qnetwork_train.state_dict(), weight_path)

    def load(self, weight_path):
        self.qnetwork_train.load_state_dict(torch.load(weight_path))
        self.qnetwork_target.load_state_dict(torch.load(weight_path))
