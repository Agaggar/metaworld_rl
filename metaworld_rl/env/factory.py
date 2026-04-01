"""Build vectorized training environments from EnvConfig (MetaWorld or Gymnasium-Robotics)."""

from __future__ import annotations

import functools
import warnings

import gymnasium as gym
import metaworld
from gymnasium.vector import VectorEnv

from metaworld_rl.config import EnvConfig
from metaworld_rl.env.wrappers import (
    VectorActionScale,
    VectorFlattenDictObs,
    VectorObservationNormalize,
)

MT10_NUM_ENVS = 10


def _make_single_task_env(task_name: str, seed: int | None, **mw_kwargs) -> gym.Env:
    """One MT1-style env (single task family, multiple goals)."""
    return metaworld.make_mt_envs(task_name, seed=seed, num_tasks=1, **mw_kwargs)


def _make_robotics_env(env_id: str, seed: int | None, **make_kwargs) -> gym.Env:
    """Single Gymnasium-Robotics env (seed applied on first reset by vector env)."""
    return gym.make(env_id, **make_kwargs)


def make_vec_env(cfg: EnvConfig) -> VectorEnv:
    """Create a vectorized env and apply action scaling / obs norm / dict-flatten wrappers.

    For ``suite=='robotics'``, registers ``gymnasium_robotics`` and builds ``num_envs``
    copies of ``robotics_env_id``, then flattens Dict observations to a Box vector.

    For MetaWorld: ``benchmark=='MT10'`` uses 10 sub-envs (``num_envs`` ignored).
    A single task name (e.g. ``reach-v3``) uses ``num_envs`` independent copies.
    """
    if cfg.suite == "robotics":
        if not cfg.robotics_env_id:
            raise ValueError("env.robotics_env_id is required when env.suite is 'robotics'")
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        make_kwargs: dict = {}
        if cfg.max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = cfg.max_episode_steps
        if cfg.render_mode is not None:
            make_kwargs["render_mode"] = cfg.render_mode
        fns = [
            functools.partial(
                _make_robotics_env,
                cfg.robotics_env_id,
                cfg.seed + i,
                **make_kwargs,
            )
            for i in range(cfg.num_envs)
        ]
        vec = gym.vector.SyncVectorEnv(fns)
        vec = VectorFlattenDictObs(vec)
        assert isinstance(vec, VectorEnv)
        if cfg.action_scale != 1.0:
            vec = VectorActionScale(vec, cfg.action_scale)
        if cfg.normalize_observations:
            vec = VectorObservationNormalize(vec)
        return vec

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
