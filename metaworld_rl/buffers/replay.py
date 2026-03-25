"""Uniform replay buffer for off-policy algorithms (SAC)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add transitions from a vector env step: shapes (n_envs, ...)."""
        n = obs.shape[0]
        for i in range(n):
            self.obs[self.ptr] = obs[i]
            self.actions[self.ptr] = actions[i]
            self.rewards[self.ptr, 0] = rewards[i]
            self.next_obs[self.ptr] = next_obs[i]
            self.dones[self.ptr, 0] = float(dones[i])
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
        )
