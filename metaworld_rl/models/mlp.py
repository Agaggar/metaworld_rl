"""MLP backbones, Gaussian policy, Q-networks, and PPO actor-critic."""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: Literal["relu", "tanh"] = "relu",
    output_dim: int | None = None,
) -> nn.Module:
    act = nn.ReLU if activation == "relu" else nn.Tanh
    layers: list[nn.Module] = []
    d = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), act()]
        d = h
    if output_dim is not None:
        layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)


def mlp_factory(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: Literal["relu", "tanh"] = "relu",
) -> nn.Module:
    """Trunk with no output head (used for shared representation)."""
    return build_mlp(input_dim, hidden_dims, activation, output_dim=None)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with state-dependent log_std; actions squashed with tanh."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: Literal["relu", "tanh"] = "relu",
    ) -> None:
        super().__init__()
        self.trunk = mlp_factory(obs_dim, hidden_dims, activation)
        h = hidden_dims[-1]
        self.mean = nn.Linear(h, act_dim)
        self.log_std = nn.Linear(h, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(obs)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns action in [-1, 1], log_prob sum over dims, and tanh-pre-activation for Q."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        log_prob = dist.log_prob(x_t).sum(-1, keepdim=True)
        y_t = torch.tanh(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(-1, keepdim=True)
        return y_t, log_prob, x_t

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """Q(s, a) with concatenated state-action input."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: Literal["relu", "tanh"] = "relu",
    ) -> None:
        super().__init__()
        self.net = build_mlp(
            obs_dim + act_dim, hidden_dims, activation, output_dim=1
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class SharedActorCritic(nn.Module):
    """Shared trunk + policy head (Gaussian) + value head (PPO)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: Literal["relu", "tanh"] = "relu",
    ) -> None:
        super().__init__()
        self.trunk = mlp_factory(obs_dim, hidden_dims, activation)
        h = hidden_dims[-1]
        self.pi_mean = nn.Linear(h, act_dim)
        self.pi_log_std = nn.Parameter(torch.zeros(1, act_dim))
        self.v = nn.Linear(h, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.trunk(obs)
        mean = self.pi_mean(x)
        log_std = self.pi_log_std.expand_as(mean).clamp(-20, 2)
        v = self.v(x)
        return mean, log_std, v.squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, v = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = dist.rsample()
        log_prob = dist.log_prob(x_t).sum(-1)
        action = torch.tanh(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        # v is already (B,) from forward(); do not squeeze again or B==1 becomes a scalar.
        return action, log_prob, entropy, v

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-prob for squashed actions (inverse tanh) + pre-tanh entropy + value."""
        mean, log_std, v = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        eps = 1e-6
        clipped = torch.clamp(actions, -1 + eps, 1 - eps)
        x_t = torch.atanh(clipped)
        log_prob = dist.log_prob(x_t).sum(-1)
        log_prob -= torch.log(1 - actions.pow(2) + eps).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, v
