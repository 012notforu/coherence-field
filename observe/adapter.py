from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from engine.params import SimulationConfig


@dataclass
class FieldAdapter:
    config: SimulationConfig
    eps: float = 1e-6
    mean: float = 0.0
    var: float = 1.0
    ema_tau: float = 200.0

    def stack(self, theta: np.ndarray, theta_dot: np.ndarray, heterogeneity: Optional[np.ndarray] = None) -> np.ndarray:
        heterogeneity = heterogeneity if heterogeneity is not None else np.zeros_like(theta)
        stacked = np.stack([theta, theta_dot, heterogeneity], axis=0)
        return stacked

    def flatten(self, stacked: np.ndarray) -> np.ndarray:
        return stacked.reshape(stacked.shape[0], -1)

    def normalize(self, data: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + self.eps)
        return (data - self.mean) / std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + self.eps)
        return data * std + self.mean

    def update_stats(self, data: np.ndarray) -> None:
        flat = data.ravel()
        mu = flat.mean()
        var = flat.var()
        decay = np.exp(-1.0 / max(self.ema_tau, 1.0))
        self.mean = decay * self.mean + (1.0 - decay) * mu
        self.var = decay * self.var + (1.0 - decay) * var + self.eps

    def prepare_observation(
        self,
        theta: np.ndarray,
        theta_dot: np.ndarray,
        heterogeneity: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        stacked = self.stack(theta, theta_dot, heterogeneity)
        batch_mean = stacked.mean()
        batch_var = stacked.var()
        normed = (stacked - batch_mean) / np.sqrt(batch_var + self.eps)
        self.update_stats(stacked)
        return {
            "raw": stacked,
            "normalized": normed,
            "flat": self.flatten(normed),
        }
