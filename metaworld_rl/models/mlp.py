"""MLP backbones, Gaussian policy, Q-networks, and PPO actor-critic."""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def init_orthogonal_linear(
    layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0
) -> nn.Linear:
    """Orthogonal weight init + constant bias init."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


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


class DiscreteQNetwork(nn.Module):
    """Q(s, .) that outputs one value per discrete action."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: Literal["relu", "tanh"] = "relu",
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, activation, output_dim=act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DiscretePolicy(nn.Module):
    """Categorical policy over discrete actions."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: Literal["relu", "tanh"] = "relu",
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, activation, output_dim=act_dim)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.logits(obs))

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob

    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.logits(obs)
        return torch.argmax(logits, dim=-1)

    def action_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class SharedActorCritic(nn.Module):
    """Shared trunk + policy head (Gaussian) + value head (PPO)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
        activation: Literal["relu", "tanh"] = "relu",
        action_space_type: Literal["continuous", "discrete"] = "continuous",
    ) -> None:
        super().__init__()
        self.pi_trunk = mlp_factory(obs_dim, hidden_dims, activation)
        self.v_trunk = mlp_factory(obs_dim, hidden_dims, activation)
        h = hidden_dims[-1]
        self.action_space_type = action_space_type
        if action_space_type == "continuous":
            self.pi_mean = init_orthogonal_linear(nn.Linear(h, act_dim), std=0.01)
            self.pi_log_std = nn.Parameter(torch.zeros(1, act_dim))
        else:
            self.pi_logits = init_orthogonal_linear(nn.Linear(h, act_dim), std=0.01)
        self.v = init_orthogonal_linear(nn.Linear(h, 1), std=1.0)
        self._init_ppo_trunks()

    def _init_ppo_trunks(self) -> None:
        # Match PPO implementation details: orthogonal init and constant bias.
        for trunk in (self.pi_trunk, self.v_trunk):
            for module in trunk.modules():
                if isinstance(module, nn.Linear):
                    init_orthogonal_linear(module, std=(2**0.5), bias_const=0.0)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        pi_x = self.pi_trunk(obs)
        v_x = self.v_trunk(obs)
        if self.action_space_type == "continuous":
            mean = self.pi_mean(pi_x)
            log_std = self.pi_log_std.expand_as(mean).clamp(-20, 2)
        else:
            mean = self.pi_logits(pi_x)
            log_std = None
        v = self.v(v_x)
        return mean, log_std, v.squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_or_logits, log_std, v = self.forward(obs)
        if self.action_space_type == "continuous":
            assert log_std is not None
            std = log_std.exp()
            dist = Normal(mean_or_logits, std)
            if deterministic:
                x_t = mean_or_logits
            else:
                x_t = dist.rsample()
            log_prob = dist.log_prob(x_t).sum(-1)
            action = torch.tanh(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            dist = Categorical(logits=mean_or_logits)
            if deterministic:
                action = torch.argmax(mean_or_logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        # v is already (B,) from forward(); do not squeeze again or B==1 becomes a scalar.
        return action, log_prob, entropy, v

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-prob for squashed actions (inverse tanh) + pre-tanh entropy + value."""
        mean_or_logits, log_std, v = self.forward(obs)
        if self.action_space_type == "continuous":
            assert log_std is not None
            std = log_std.exp()
            dist = Normal(mean_or_logits, std)
            eps = 1e-6
            clipped = torch.clamp(actions, -1 + eps, 1 - eps)
            x_t = torch.atanh(clipped)
            log_prob = dist.log_prob(x_t).sum(-1)
            log_prob -= torch.log(1 - actions.pow(2) + eps).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            dist = Categorical(logits=mean_or_logits)
            discrete_actions = actions.long().view(-1)
            log_prob = dist.log_prob(discrete_actions)
            entropy = dist.entropy()
        return log_prob, entropy, v
