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
        self.next_values = np.zeros((n,), dtype=np.float32)
        self.delta_ts = np.zeros((n,), dtype=np.float32)
        self.env_ids = np.zeros((n,), dtype=np.int32)
        self.pos = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_values: np.ndarray,
        delta_ts: np.ndarray,
        env_ids: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        sl = slice(self.pos, self.pos + n)
        self.obs[sl] = obs
        self.actions[sl] = actions
        self.log_probs[sl] = log_probs
        self.rewards[sl] = rewards
        self.values[sl] = values
        self.dones[sl] = dones
        self.next_values[sl] = next_values
        self.delta_ts[sl] = delta_ts
        self.env_ids[sl] = env_ids
        self.pos += n

    def full(self) -> bool:
        return self.pos >= self.n_steps * self.num_envs

    def reset_storage(self) -> None:
        self.pos = 0

    def compute_returns(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE-Lambda with variable transition spans (delta_ts)."""
        n = self.pos
        advantages = np.zeros((n,), dtype=np.float32)
        env_last_gae = np.zeros((self.num_envs,), dtype=np.float32)
        for idx in range(n - 1, -1, -1):
            env_id = int(self.env_ids[idx])
            dt = float(self.delta_ts[idx])
            non_terminal = 1.0 - self.dones[idx]
            gamma_pow = gamma**dt
            gl_pow = (gamma * gae_lambda) ** dt
            delta = (
                self.rewards[idx]
                + gamma_pow * self.next_values[idx] * non_terminal
                - self.values[idx]
            )
            env_last_gae[env_id] = delta + gl_pow * non_terminal * env_last_gae[env_id]
            advantages[idx] = env_last_gae[env_id]

        adv_flat = advantages
        ret_flat = adv_flat + self.values[:n]
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
