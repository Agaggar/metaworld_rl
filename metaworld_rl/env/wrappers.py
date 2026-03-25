"""Vector-env wrappers: frame skip, action scaling, observation normalization."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from gymnasium.vector import VectorEnv, VectorWrapper


class RunningMeanStd:
    """Welford-style running statistics for vector observations."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: npt.NDArray[np.floating]) -> None:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        new_var = m2 / tot
        self.mean = new_mean
        self.var = new_var
        self.count = tot

    def normalize(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


class VectorFrameSkip(VectorWrapper):
    """Repeat the same action for `skip` env steps; rewards are summed."""

    def __init__(self, env: VectorEnv, skip: int) -> None:
        if skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self._skip = skip
        super().__init__(env)

    def step(self, actions):
        obs, r, term, trunc, infos = self.env.step(actions)
        total_r = r.astype(np.float64, copy=True)
        for _ in range(self._skip - 1):
            if np.any(term) or np.any(trunc):
                break
            obs, r, term, trunc, infos = self.env.step(actions)
            total_r += r
        return obs, total_r.astype(np.float32), term, trunc, infos


class VectorActionScale(VectorWrapper):
    """Scale actions before the env; result is clipped to [-1, 1] (lower scale => slower motion)."""

    def __init__(self, env: VectorEnv, scale: float) -> None:
        if scale <= 0:
            raise ValueError("action_scale must be > 0")
        self._scale = float(scale)
        super().__init__(env)

    def step(self, actions):
        a = np.clip(actions * self._scale, -1.0, 1.0)
        return self.env.step(a)


class VectorObservationNormalize(VectorWrapper):
    """Running mean/var normalization of observations (updates when training=True)."""

    def __init__(self, env: VectorEnv, epsilon: float = 1e-4) -> None:
        self._rms: RunningMeanStd | None = None
        self._epsilon = epsilon
        self.training = True
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        if self._rms is None:
            self._rms = RunningMeanStd(obs.shape[1:], epsilon=self._epsilon)
        if self.training:
            self._rms.update(obs)
        return self._norm(obs), infos

    def step(self, actions):
        obs, r, term, trunc, infos = self.env.step(actions)
        assert self._rms is not None
        if self.training:
            self._rms.update(obs)
        return self._norm(obs), r, term, trunc, infos

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        assert self._rms is not None
        return self._rms.normalize(obs)
