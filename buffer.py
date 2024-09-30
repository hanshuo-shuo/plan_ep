import numpy as np
import random
import torch
from collections import deque, namedtuple


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, type='random'):
        if type == 'random':
            """Randomly sample a batch of experiences from memory."""
            experiences = random.choices(self.memory, k=self.batch_size)
            states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
                self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
                self.device)
        elif type == 'bias':
            """"Sample a batch of experiences from memory with bias towards the ones with negative rewards."""
            negative_experiences = [e for e in self.memory if e.reward < 0]
            # sample 1/3 of the batch from negative experiences and make sure replace=False
            negative_experiences = random.choices(negative_experiences, k=self.batch_size // 2)
            other_experiences = random.choices(self.memory, k=self.batch_size - len(negative_experiences))
            experiences = negative_experiences + other_experiences
            states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
                self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
                self.device)
        else:
            raise ValueError('Invalid type')
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)