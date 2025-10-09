from __future__ import annotations

from typing import Optional

import numpy as np

from .params import SchedulerParams, GridSpec

Array = np.ndarray


class AsyncScheduler:
    def __init__(self, params: SchedulerParams, grid: GridSpec, seed: int) -> None:
        self.params = params
        self.grid = grid
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def sample_mask(self, dt: float) -> Array:
        if self.params.mode == "poisson":
            return self._poisson_mask(dt)
        if self.params.mode == "shuffle":
            return self._shuffle_mask()
        raise ValueError(f"Unknown scheduler mode: {self.params.mode}")

    def _poisson_mask(self, dt: float) -> Array:
        rate = max(self.params.poisson_rate, 0.0)
        lam = min(rate * dt, 1.0)
        jitter = 1.0 + self.params.jitter * self._rng.normal()
        lam = np.clip(lam * max(jitter, 0.0), 0.0, 1.0)
        mask = self._rng.random(self.grid.shape) < lam
        self._bump()
        return mask

    def _shuffle_mask(self) -> Array:
        flat = np.arange(self.grid.size)
        self._rng.shuffle(flat)
        mask = np.zeros(self.grid.size, dtype=bool)
        fraction = np.clip(self.params.poisson_rate, 0.0, 1.0)
        count = int(fraction * self.grid.size)
        mask[:count] = True
        mask = mask.reshape(self.grid.shape)
        self._bump()
        return mask

    def _bump(self) -> None:
        self._step += 1
        if self.params.reseed_every > 0 and self._step % self.params.reseed_every == 0:
            self._rng = np.random.default_rng(self._rng.integers(0, 2**32 - 1))

    def ks_statistic(self, samples: np.ndarray, dt: float) -> float:
        if len(samples) == 0:
            return 0.0
        rate = max(self.params.poisson_rate, 1e-8)
        expected = 1.0 - np.exp(-rate * dt)
        empirical = np.mean(samples)
        return abs(empirical - expected)
