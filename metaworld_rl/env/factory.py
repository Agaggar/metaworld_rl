"""Build vectorized MetaWorld training environments from EnvConfig."""

from __future__ import annotations

import functools
import warnings

import gymnasium as gym
import metaworld
from gymnasium.vector import VectorEnv

from metaworld_rl.config import EnvConfig
from metaworld_rl.env.wrappers import (
    VectorActionScale,
    VectorObservationNormalize,
)

MT10_NUM_ENVS = 10


def _make_single_task_env(task_name: str, seed: int | None, **mw_kwargs) -> gym.Env:
    """One MT1-style env (single task family, multiple goals)."""
    return metaworld.make_mt_envs(task_name, seed=seed, num_tasks=1, **mw_kwargs)


def make_vec_env(cfg: EnvConfig) -> VectorEnv:
    """Create a vectorized MetaWorld env and apply action scaling / obs norm wrappers.

    For ``benchmark=='MT10'``, the underlying vector env always has 10 sub-envs (one per task).
    ``num_envs`` is ignored in that case (a warning is emitted if it is not 10).
    For a single task name (e.g. ``reach-v3``), ``num_envs`` independent copies are created.
    """
    mw_kwargs: dict = {}
    if cfg.max_episode_steps is not None:
        mw_kwargs["max_episode_steps"] = cfg.max_episode_steps
    if cfg.render_mode is not None:
        mw_kwargs["render_mode"] = cfg.render_mode
    if cfg.use_one_hot_task_id:
        mw_kwargs["use_one_hot"] = True
    if cfg.camera_name is not None:
        mw_kwargs["camera_name"] = cfg.camera_name

    if cfg.benchmark.upper() == "MT10":
        if cfg.num_envs != MT10_NUM_ENVS:
            warnings.warn(
                f"MT10 uses exactly {MT10_NUM_ENVS} parallel envs; ignoring num_envs={cfg.num_envs}.",
                stacklevel=2,
            )
        vec = metaworld.make_mt_envs(
            "MT10",
            seed=cfg.seed,
            vector_strategy="sync",
            **mw_kwargs,
        )
    else:
        task = cfg.benchmark
        fns = [
            functools.partial(
                _make_single_task_env,
                task,
                cfg.seed + i,
                **mw_kwargs,
            )
            for i in range(cfg.num_envs)
        ]
        vec = gym.vector.SyncVectorEnv(fns)

    assert isinstance(vec, VectorEnv)

    if cfg.action_scale != 1.0:
        vec = VectorActionScale(vec, cfg.action_scale)
    if cfg.normalize_observations:
        vec = VectorObservationNormalize(vec)

    return vec
