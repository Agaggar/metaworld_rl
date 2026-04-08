"""Soft Actor-Critic (from scratch, PyTorch)."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim

from metaworld_rl.config import ModelConfig, SacConfig
from metaworld_rl.models.mlp import (
    DiscretePolicy,
    DiscreteQNetwork,
    GaussianPolicy,
    QNetwork,
)


class SacAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        model: ModelConfig,
        sac: SacConfig,
        device: torch.device,
        action_space_type: str,
    ) -> None:
        self.device = device
        self.action_space_type = action_space_type
        self.gamma = sac.gamma
        self.tau = sac.tau
        self.auto_alpha = sac.auto_alpha
        self.policy_frequency = max(1, sac.policy_frequency)
        self.target_network_frequency = max(1, sac.target_network_frequency)
        self.update_step = 0
        self.target_entropy = sac.target_entropy
        if self.target_entropy is None:
            if self.action_space_type == "continuous":
                self.target_entropy = -float(act_dim)
            else:
                self.target_entropy = -float(torch.log(torch.tensor(act_dim)).item())

        if self.action_space_type == "continuous":
            self.actor = GaussianPolicy(
                obs_dim, act_dim, model.hidden_dims, model.activation
            ).to(device)
            self.q1 = QNetwork(obs_dim, act_dim, model.hidden_dims, model.activation).to(
                device
            )
            self.q2 = QNetwork(obs_dim, act_dim, model.hidden_dims, model.activation).to(
                device
            )
        else:
            self.actor = DiscretePolicy(
                obs_dim, act_dim, model.hidden_dims, model.activation
            ).to(device)
            self.q1 = DiscreteQNetwork(
                obs_dim, act_dim, model.hidden_dims, model.activation
            ).to(device)
            self.q2 = DiscreteQNetwork(
                obs_dim, act_dim, model.hidden_dims, model.activation
            ).to(device)
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)
        for p in self.q1_t.parameters():
            p.requires_grad = False
        for p in self.q2_t.parameters():
            p.requires_grad = False

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=sac.lr)
        self.q_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=sac.lr
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = sac.alpha
        self.alpha_opt = optim.Adam([self.log_alpha], lr=sac.lr) if self.auto_alpha else None

    @property
    def alpha_value(self) -> float:
        if self.auto_alpha:
            return float(self.log_alpha.exp().item())
        return float(self.alpha)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """obs shape (batch, obs_dim); returns actions in [-1,1]."""
        with torch.no_grad():
            if self.action_space_type == "continuous":
                if deterministic:
                    return self.actor.mean_action(obs)
                a, _, _ = self.actor.sample(obs)
                return a
            if deterministic:
                return self.actor.greedy_action(obs)
            a, _ = self.actor.sample(obs)
            return a

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        self.update_step += 1
        obs, actions, rewards, next_obs, dones, discounts = batch
        alpha = self.log_alpha.exp() if self.auto_alpha else torch.tensor(
            self.alpha, device=self.device
        )

        if self.action_space_type == "continuous":
            with torch.no_grad():
                next_a, next_log_pi, _ = self.actor.sample(next_obs)
                q1n = self.q1_t(next_obs, next_a)
                q2n = self.q2_t(next_obs, next_a)
                qn = torch.min(q1n, q2n)
                target_q = rewards + (1 - dones) * discounts * (qn - alpha * next_log_pi)

            q1 = self.q1(obs, actions)
            q2 = self.q2(obs, actions)
            q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        else:
            act_idx = actions.long().view(-1, 1)
            with torch.no_grad():
                next_probs, next_log_probs = self.actor.action_probs(next_obs)
                q1n = self.q1_t(next_obs)
                q2n = self.q2_t(next_obs)
                min_qn = torch.min(q1n, q2n)
                v_next = (next_probs * (min_qn - alpha * next_log_probs)).sum(
                    dim=-1, keepdim=True
                )
                target_q = rewards + (1 - dones) * discounts * v_next

            q1_all = self.q1(obs)
            q2_all = self.q2(obs)
            q1 = q1_all.gather(1, act_idx)
            q2 = q2_all.gather(1, act_idx)
            q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        log_pi = torch.zeros((obs.shape[0], 1), device=self.device)
        if self.update_step % self.policy_frequency == 0:
            if self.action_space_type == "continuous":
                a_pi, log_pi, _ = self.actor.sample(obs)
                q1_pi = self.q1(obs, a_pi)
                q2_pi = self.q2(obs, a_pi)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha * log_pi - q_pi).mean()
            else:
                probs, log_probs = self.actor.action_probs(obs)
                min_q = torch.min(self.q1(obs), self.q2(obs))
                actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()
                log_pi = (probs * log_probs).sum(dim=-1, keepdim=True)

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if (
            self.auto_alpha
            and self.alpha_opt is not None
            and self.update_step % self.policy_frequency == 0
        ):
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        if self.update_step % self.target_network_frequency == 0:
            with torch.no_grad():
                for p, pt in zip(self.q1.parameters(), self.q1_t.parameters(), strict=True):
                    pt.data.mul_(1 - self.tau).add_(p.data * self.tau)
                for p, pt in zip(self.q2.parameters(), self.q2_t.parameters(), strict=True):
                    pt.data.mul_(1 - self.tau).add_(p.data * self.tau)

        return {
            "loss_critic": float(q_loss.item()),
            "loss_actor": float(actor_loss.item()),
            "alpha": self.alpha_value,
            "q_gap": float((q1 - q2).abs().mean().item()),
            "loss_alpha": float(alpha_loss.item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(),
            "log_alpha": self.log_alpha.detach(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.q1_t.load_state_dict(sd["q1_t"])
        self.q2_t.load_state_dict(sd["q2_t"])
        if "log_alpha" in sd:
            self.log_alpha.data.copy_(sd["log_alpha"])
