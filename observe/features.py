from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from engine.energy import total_energy_density
from engine.ops import laplacian, norm_sq_grad
from engine.params import SimulationConfig


@dataclass
class RunningStandardizer:
    """Maintains exponential moving statistics for feature normalization."""

    dim: int
    momentum: float = 0.05
    eps: float = 1e-6
    mean: np.ndarray = field(init=False, repr=False)
    second_moment: np.ndarray = field(init=False, repr=False)
    var: np.ndarray = field(init=False, repr=False)
    count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.second_moment = np.ones(self.dim, dtype=np.float32)
        self.var = np.ones(self.dim, dtype=np.float32)

    def reset(self) -> None:
        self.count = 0
        self.mean.fill(0.0)
        self.second_moment.fill(1.0)
        self.var.fill(1.0)

    def update(self, sample: np.ndarray) -> None:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape != (self.dim,):
            raise ValueError(f"Expected sample with shape ({self.dim},), got {sample.shape}")
        self.count += 1
        rate = max(self.momentum, 1.0 / float(self.count))
        if self.count == 1:
            self.mean = sample.copy()
            self.second_moment = sample ** 2
        else:
            self.mean = (1.0 - rate) * self.mean + rate * sample
            self.second_moment = (1.0 - rate) * self.second_moment + rate * (sample ** 2)
        self.var = np.maximum(self.second_moment - self.mean ** 2, self.eps)

    def transform(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape != (self.dim,):
            raise ValueError(f"Expected sample with shape ({self.dim},), got {sample.shape}")
        std = np.sqrt(self.var + self.eps)
        return (sample - self.mean) / std

    def inverse_transform(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape != (self.dim,):
            raise ValueError(f"Expected sample with shape ({self.dim},), got {sample.shape}")
        std = np.sqrt(self.var + self.eps)
        return sample * std + self.mean


@dataclass
class FeatureVector:
    raw: np.ndarray
    normalized: np.ndarray


class CartPoleFeatureExtractor:
    """Physics-aware feature extractor for SCFD-driven cart-pole control."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        momentum: float = 0.05,
        standardize: bool = True,
        deadzone_scale: Sequence[float] | None = None,
    ) -> None:
        self.config = config
        self.dx = config.grid.spacing
        self.mid_col = config.grid.shape[1] // 2
        self.standardize = standardize
        self.feature_names: tuple[str, ...] = (
            "delta_energy",
            "delta_edge",
            "delta_curvature",
            "energy_mean",
            "theta",
            "theta_dot",
            "cart_position",
            "cart_velocity",
            "prev_action",
            "prev_delta_energy",
        )
        self._stats = RunningStandardizer(dim=len(self.feature_names), momentum=momentum)
        self._prev_delta_energy: float = 0.0
        self._prev_action: float = 0.0
        if deadzone_scale is None:
            deadzone_scale = [0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5]
        if len(deadzone_scale) != len(self.feature_names):
            raise ValueError("deadzone_scale must match feature dimension")
        self.deadzone_scale = np.asarray(deadzone_scale, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def reset(self) -> None:
        self._stats.reset()
        self._prev_delta_energy = 0.0
        self._prev_action = 0.0

    def _split_sides(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = array[:, : self.mid_col]
        right = array[:, self.mid_col :]
        return left, right

    def _energy_features(self, theta: np.ndarray, theta_dot: np.ndarray) -> tuple[float, float, float, float]:
        energy = total_energy_density(theta, theta_dot, self.config.physics)
        grad = np.sqrt(np.maximum(norm_sq_grad(theta, dx=self.dx), 0.0))
        curvature = np.abs(laplacian(theta, dx=self.dx))
        left_e, right_e = self._split_sides(energy)
        left_g, right_g = self._split_sides(grad)
        left_c, right_c = self._split_sides(curvature)
        delta_energy = float(right_e.mean() - left_e.mean())
        delta_grad = float(right_g.mean() - left_g.mean())
        delta_curv = float(right_c.mean() - left_c.mean())
        energy_mean = float(energy.mean())
        return delta_energy, delta_grad, delta_curv, energy_mean

    def extract(
        self,
        theta: np.ndarray,
        theta_dot: np.ndarray,
        env_state: np.ndarray,
        *,
        prev_action: float | None = None,
    ) -> FeatureVector:
        env = np.asarray(env_state, dtype=np.float32)
        if env.shape != (4,):
            raise ValueError(f"Expected env_state with shape (4,), got {env.shape}")
        if prev_action is None:
            prev_action = self._prev_action
        delta_energy, delta_grad, delta_curv, energy_mean = self._energy_features(theta, theta_dot)
        raw = np.array(
            [
                delta_energy,
                delta_grad,
                delta_curv,
                energy_mean,
                float(env[2]),
                float(env[3]),
                float(env[0]),
                float(env[1]),
                float(prev_action),
                float(self._prev_delta_energy),
            ],
            dtype=np.float32,
        )
        self._stats.update(raw)
        normalized = self._stats.transform(raw) if self.standardize else raw.copy()
        self._prev_delta_energy = delta_energy
        self._prev_action = float(prev_action)
        return FeatureVector(raw=raw, normalized=normalized)

    def deadzone_scaled(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (self.dimension,):
            raise ValueError(f"Expected feature vector with shape ({self.dimension},), got {features.shape}")
        return features * self.deadzone_scale
