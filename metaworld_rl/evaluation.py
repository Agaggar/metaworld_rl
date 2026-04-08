"""Evaluation metrics and optional rollout videos."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v2 as imageio
import numpy as np
import torch

if TYPE_CHECKING:
    from metaworld_rl.agents.ppo import PpoAgent
    from metaworld_rl.agents.sac import SacAgent


def _success_from_vector_infos(infos: dict[str, Any]) -> np.ndarray | None:
    """Batch success signal from Gymnasium vector ``infos`` (MetaWorld vs Robotics keys)."""
    if "success" in infos:
        s = infos["success"]
    elif "is_success" in infos:
        s = infos["is_success"]
    elif "_is_success" in infos:
        s = infos["_is_success"]
    else:
        return None
    if isinstance(s, np.ndarray):
        return s.astype(np.float64, copy=False)
    return None


def set_obs_norm_training(env: Any, training: bool) -> None:
    """If wrapped with VectorObservationNormalize, toggle training (freeze stats during eval)."""
    cur = env
    for _ in range(20):
        if hasattr(cur, "training"):
            cur.training = training
            return
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break


def evaluate_vector_env(
    env,
    agent: "SacAgent | PpoAgent",
    device: torch.device,
    num_envs: int,
    max_steps: int = 500,
) -> dict[str, float]:
    """Run parallel envs until each completes one episode or hits max_steps; report success rate and mean return."""
    set_obs_norm_training(env, False)
    obs, _ = env.reset(seed=None)
    returns = np.zeros(num_envs, dtype=np.float64)
    success = np.zeros(num_envs, dtype=np.float64)
    active = np.ones(num_envs, dtype=bool)
    steps = 0

    def _actions_from_agent(obs_t: torch.Tensor) -> torch.Tensor:
        """Extract actions from agent.act() outputs.

        SAC returns actions tensor.
        PPO returns (actions, log_probs, values).
        """
        out = agent.act(obs_t, deterministic=True)
        if isinstance(out, tuple):
            return out[0]
        return out

    while active.any() and steps < max_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions = _actions_from_agent(obs_t).cpu().numpy()

        obs, rewards, term, trunc, infos = env.step(actions)
        returns += rewards * active.astype(np.float64)
        done = term | trunc
        succ = _success_from_vector_infos(infos)
        if succ is not None:
            success = np.maximum(success, succ * active.astype(np.float64))
        active &= ~done
        steps += 1

    set_obs_norm_training(env, True)
    return {
        "success_rate": float(success.mean()), # if num_envs else 0.0,
        "episode_return_mean": float(returns.mean()),
    }


def record_video(
    env,
    agent: "SacAgent | PpoAgent",
    device: torch.device,
    path: str | Path,
    max_steps: int = 200,
    fps: int = 20,
) -> None:
    """Save a short rgb_array rollout from the first sub-env (requires render_mode='rgb_array' on base env)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    set_obs_norm_training(env, False)
    obs, _ = env.reset(seed=0)
    frames: list[np.ndarray] = []

    def _actions_from_agent(obs_t: torch.Tensor) -> torch.Tensor:
        out = agent.act(obs_t, deterministic=True)
        if isinstance(out, tuple):
            return out[0]
        return out

    for _ in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions = _actions_from_agent(obs_t).cpu().numpy()

        obs, _, term, trunc, _ = env.step(actions)
        try:
            frame = env.render()
        except Exception:
            set_obs_norm_training(env, True)
            return
        if frame is None:
            break
        if isinstance(frame, (tuple, list)):
            frame = frame[0]
        if hasattr(frame, "shape") and len(frame.shape) == 4:
            frame = frame[0]
        frames.append(np.asarray(frame))
        if term[0] or trunc[0]:
            break

    set_obs_norm_training(env, True)
    if frames:
        imageio.mimsave(str(path), frames, fps=fps)
