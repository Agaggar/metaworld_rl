"""On-policy rollout storage for PPO."""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.device = device
        n = n_steps * num_envs
        self.obs = np.zeros((n, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((n,), dtype=np.float32)
        self.rewards = np.zeros((n,), dtype=np.float32)
        self.values = np.zeros((n,), dtype=np.float32)
        self.dones = np.zeros((n,), dtype=np.float32)
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        sl = slice(self.pos, self.pos + n)
        self.obs[sl] = obs
        self.actions[sl] = actions
        self.log_probs[sl] = log_probs
        self.rewards[sl] = rewards
        self.values[sl] = values
        self.dones[sl] = dones
        self.pos += n

    def full(self) -> bool:
        return self.pos >= self.n_steps * self.num_envs

    def reset_storage(self) -> None:
        self.pos = 0

    def compute_returns(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE-Lambda advantages and returns. last_values: (num_envs,) bootstrap after last step."""
        T, N = self.n_steps, self.num_envs
        rewards = self.rewards.reshape(T, N)
        values = self.values.reshape(T, N)
        dones = self.dones.reshape(T, N)
        advantages = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_v = last_values
            else:
                next_v = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        adv_flat = advantages.reshape(-1)
        ret_flat = adv_flat + self.values
        return adv_flat, ret_flat

    def batches(self, minibatch_size: int, advantages: np.ndarray, returns: np.ndarray):
        n = self.n_steps * self.num_envs
        idx = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            b = idx[start : start + minibatch_size]
            yield (
                torch.as_tensor(self.obs[b], device=self.device),
                torch.as_tensor(self.actions[b], device=self.device),
                torch.as_tensor(self.log_probs[b], device=self.device),
                torch.as_tensor(advantages[b], device=self.device),
                torch.as_tensor(returns[b], device=self.device),
            )
