"""Proximal Policy Optimization (GAE, clipped objective)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim

from metaworld_rl.config import ModelConfig, PpoConfig
from metaworld_rl.models.mlp import SharedActorCritic


class PpoAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        model: ModelConfig,
        ppo: PpoConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.ppo = ppo
        self.net = SharedActorCritic(
            obs_dim, act_dim, model.hidden_dims, model.activation
        ).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=ppo.lr, eps=1e-5)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns action, log_prob, value."""
        with torch.no_grad():
            return self.net.get_action_and_value(obs, deterministic=deterministic)

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        n_updates = 0

        n = obs.shape[0]
        mb = min(self.ppo.minibatch_size, n)
        for _ in range(self.ppo.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb):
                idx = perm[start : start + mb]
                o = obs[idx]
                a = actions[idx]
                old_lp = old_log_probs[idx]
                adv = advantages[idx]
                ret = returns[idx]

                log_probs, entropy, values = self.net.evaluate_actions(o, a)
                ratio = torch.exp(log_probs - old_lp)
                surr1 = ratio * adv
                surr2 = (
                    torch.clamp(ratio, 1 - self.ppo.clip_range, 1 + self.ppo.clip_range)
                    * adv
                )
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(values, ret)

                loss = (
                    pi_loss
                    + self.ppo.vf_coef * v_loss
                    - self.ppo.ent_coef * entropy.mean()
                )

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.ppo.max_grad_norm
                )
                self.opt.step()

                total_pi_loss += float(pi_loss.item())
                total_v_loss += float(v_loss.item())
                total_ent += float(entropy.mean().item())
                n_updates += 1

        return {
            "loss_policy": total_pi_loss / max(n_updates, 1),
            "loss_value": total_v_loss / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"net": self.net.state_dict()}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.net.load_state_dict(sd["net"])
