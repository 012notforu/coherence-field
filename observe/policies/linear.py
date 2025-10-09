from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class LinearPolicyConfig:
    """Configuration parameters for the linear CartPole policy."""

    action_clip: float = 10.0
    action_delta_clip: float = 2.0
    smooth_lambda: float = 0.25


class LinearPolicy:
    """Smoothed, clamped linear policy over standardized feature vectors."""

    def __init__(
        self,
        dim: int,
        *,
        config: LinearPolicyConfig | None = None,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
    ) -> None:
        self.config = config or LinearPolicyConfig()
        self.weights = (
            np.asarray(weights, dtype=np.float32)
            if weights is not None
            else np.zeros(dim, dtype=np.float32)
        )
        if self.weights.shape != (dim,):
            raise ValueError(f"Expected weights with shape ({dim},), got {self.weights.shape}")
        self.bias = float(bias)
        self.prev_action: float = 0.0

    @property
    def dim(self) -> int:
        return self.weights.shape[0]

    def reset(self) -> None:
        self.prev_action = 0.0

    def set_parameters(self, weights: Sequence[float], bias: float | None = None) -> None:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != self.weights.shape:
            raise ValueError(f"Expected weights with shape {self.weights.shape}, got {weights.shape}")
        self.weights = weights
        if bias is not None:
            self.bias = float(bias)

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.weights, np.array([self.bias], dtype=np.float32)])

    def load_vector(self, vector: Sequence[float]) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape[0] != self.dim + 1:
            raise ValueError(f"Expected vector of length {self.dim + 1}, got {vector.shape[0]}")
        self.weights = vector[:-1]
        self.bias = float(vector[-1])

    def act(
        self,
        features: Sequence[float],
        *,
        deadzone: bool = False,
        deadzone_scale: Sequence[float] | None = None,
    ) -> Tuple[float, dict[str, float]]:
        feats = np.asarray(features, dtype=np.float32)
        if feats.shape != self.weights.shape:
            raise ValueError(f"Expected feature vector with shape {self.weights.shape}, got {feats.shape}")
        used_feats = feats
        if deadzone:
            if deadzone_scale is not None:
                scale = np.asarray(deadzone_scale, dtype=np.float32)
                if scale.shape != feats.shape:
                    raise ValueError(f"deadzone_scale shape {scale.shape} incompatible with features {feats.shape}")
                used_feats = feats * scale
            else:
                used_feats = feats.copy()
        raw = float(np.dot(self.weights, used_feats) + self.bias)
        prev = self.prev_action
        target = (1.0 - self.config.smooth_lambda) * prev + self.config.smooth_lambda * raw
        delta = np.clip(target - prev, -self.config.action_delta_clip, self.config.action_delta_clip)
        action = float(np.clip(prev + delta, -self.config.action_clip, self.config.action_clip))
        self.prev_action = action
        info = {
            "raw": raw,
            "target": target,
            "delta": float(delta),
            "prev_action": prev,
            "action": action,
            "deadzone": bool(deadzone),
            "features": feats.copy(),
            "effective_features": used_feats.copy(),
        }
        return action, info
