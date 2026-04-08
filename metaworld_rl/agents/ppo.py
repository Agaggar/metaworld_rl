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
        action_space_type: str,
    ) -> None:
        self.device = device
        self.ppo = ppo
        self.action_space_type = action_space_type
        self.net = SharedActorCritic(
            obs_dim,
            act_dim,
            model.hidden_dims,
            model.activation,
            action_space_type=action_space_type,
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
        old_values: torch.Tensor,
    ) -> dict[str, float]:
        if self.ppo.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        n = obs.shape[0]
        mb = min(self.ppo.minibatch_size, n)
        for _ in range(self.ppo.n_epochs):
            perm = torch.randperm(n, device=self.device)
            stop_early = False
            for start in range(0, n, mb):
                idx = perm[start : start + mb]
                o = obs[idx]
                a = actions[idx]
                old_lp = old_log_probs[idx]
                adv = advantages[idx]
                ret = returns[idx]
                old_v = old_values[idx]

                log_probs, entropy, values = self.net.evaluate_actions(o, a)
                ratio = torch.exp(log_probs - old_lp)
                surr1 = ratio * adv
                surr2 = (
                    torch.clamp(ratio, 1 - self.ppo.clip_range, 1 + self.ppo.clip_range)
                    * adv
                )
                pi_loss = -torch.min(surr1, surr2).mean()

                if self.ppo.clip_vloss:
                    v_loss_unclipped = (values - ret).pow(2)
                    v_clipped = old_v + torch.clamp(
                        values - old_v, -self.ppo.clip_range, self.ppo.clip_range
                    )
                    v_loss_clipped = (v_clipped - ret).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * F.mse_loss(values, ret)

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

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (log_probs - old_lp)).mean()

                total_pi_loss += float(pi_loss.item())
                total_v_loss += float(v_loss.item())
                total_ent += float(entropy.mean().item())
                total_approx_kl += float(approx_kl.item())
                n_updates += 1

                if (
                    self.ppo.target_kl is not None
                    and float(approx_kl.item()) > self.ppo.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        return {
            "loss_policy": total_pi_loss / max(n_updates, 1),
            "loss_value": total_v_loss / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
            "approx_kl": total_approx_kl / max(n_updates, 1),
        }

    def set_lr(self, lr: float) -> None:
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def state_dict(self) -> dict[str, Any]:
        return {"net": self.net.state_dict()}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.net.load_state_dict(sd["net"])
