import torch
from collections import namedtuple, deque
import random
import numpy as np
from typing import Tuple


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, done: np.ndarray) -> None:
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experiences = random.sample(self.memory, k=self.batch_size)


        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)