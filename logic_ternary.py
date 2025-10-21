# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2025 012loop / Looptronics
"""Balanced ternary logic helpers for SCFD controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from observe.features import CartPoleFeatureExtractor, FeatureVector

TRITS = np.array([-1, 0, 1], dtype=np.int8)


def _as_int8(value: np.ndarray | int | float) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int8)
    return arr


def hysteretic_sign_trit(
    values: np.ndarray | float,
    *,
    hi: float,
    lo: float,
    previous: Optional[np.ndarray | int] = None,
) -> np.ndarray:
    """Return balanced sign trits with hysteresis around zero."""
    if hi < 0 or lo < 0 or hi < lo:
        raise ValueError("Require hi >= lo >= 0")
    x = np.asarray(values, dtype=np.float32)
    prev = np.zeros_like(x, dtype=np.int8) if previous is None else _as_int8(previous)
    if prev.shape != x.shape:
        prev = np.broadcast_to(prev, x.shape).copy()
    out = prev.copy()
    pos = out == 1
    neg = out == -1
    out[pos & (x < lo)] = 0
    out[neg & (x > -lo)] = 0
    zero_mask = out == 0
    out[zero_mask & (x > hi)] = 1
    out[zero_mask & (x < -hi)] = -1
    return out.astype(np.int8)


def bin3_trit(
    values: np.ndarray | float,
    *,
    low: float,
    high: float,
    previous: Optional[np.ndarray | int] = None,
    margin: float = 0.05,
) -> np.ndarray:
    """Balanced ternary binning with gentle hysteresis between thirds."""
    if high <= low:
        raise ValueError("high must exceed low")
    x = np.asarray(values, dtype=np.float32)
    width = float(high - low)
    t_low = low + width / 3.0
    t_high = high - width / 3.0
    raw = np.zeros_like(x, dtype=np.int8)
    raw[x < t_low] = -1
    raw[x > t_high] = 1
    if previous is None:
        return raw
    prev = _as_int8(previous)
    if prev.shape != x.shape:
        prev = np.broadcast_to(prev, x.shape)
    band = margin * width
    stay_neg = prev == -1
    stay_pos = prev == 1
    raw[stay_neg & (x < t_low + band)] = -1
    raw[stay_pos & (x > t_high - band)] = 1
    raw[(prev == 0) & (x >= t_low - band) & (x <= t_high + band)] = 0
    return raw.astype(np.int8)


def ternary_inv(trits: np.ndarray | int) -> np.ndarray:
    return (-_as_int8(trits)).astype(np.int8)


def ternary_majority(*trits: np.ndarray | int) -> np.ndarray:
    if not trits:
        raise ValueError("Provide at least one trit input")
    arrays = [_as_int8(t) for t in trits]
    stacked = np.stack(arrays, axis=0)
    pos = np.sum(stacked == 1, axis=0)
    neg = np.sum(stacked == -1, axis=0)
    result = np.zeros_like(arrays[0], dtype=np.int8)
    result[pos > neg] = 1
    result[neg > pos] = -1
    return result


def ternary_mux(selector: np.ndarray | int, pos: np.ndarray | int, neg: np.ndarray | int) -> np.ndarray:
    sel = _as_int8(selector)
    pos_arr = _as_int8(pos)
    neg_arr = _as_int8(neg)
    return np.where(sel > 0, pos_arr, np.where(sel < 0, neg_arr, 0)).astype(np.int8)

_ADD3_TABLE: Dict[Tuple[int, int], int] = {
    (-1, -1): -1,
    (-1, 0): -1,
    (-1, 1): 0,
    (0, -1): -1,
    (0, 0): 0,
    (0, 1): 1,
    (1, -1): 0,
    (1, 0): 1,
    (1, 1): 1,
}


def ternary_add3(a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray:
    arr_a = _as_int8(a)
    arr_b = _as_int8(b)
    lut = np.vectorize(lambda x, y: _ADD3_TABLE[(int(x), int(y))], otypes=[np.int8])
    return lut(arr_a, arr_b)


@dataclass
class CartPoleTernaryConfig:
    theta_hi: float = np.deg2rad(1.5)
    theta_lo: float = np.deg2rad(0.6)
    theta_dot_hi: float = 0.3
    theta_dot_lo: float = 0.1
    energy_hi: float = 0.01
    energy_lo: float = 0.005
    force_scale: float = 7.5
    smooth_lambda: float = 0.5
    action_clip: float = 10.0
    deadzone_angle: float = np.deg2rad(0.5)
    deadzone_ang_vel: float = 0.05


@dataclass
class CartPoleTernaryController:
    extractor: CartPoleFeatureExtractor
    config: CartPoleTernaryConfig = field(default_factory=CartPoleTernaryConfig)

    def __post_init__(self) -> None:
        self._prev_trits: Dict[str, int] = {"theta": 0, "thetadot": 0, "energy": 0}
        self.prev_action: float = 0.0
        self.last_trits: Dict[str, int] = {}
        self.last_raw_features: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.extractor.reset()
        self._prev_trits = {"theta": 0, "thetadot": 0, "energy": 0}
        self.prev_action = 0.0
        self.last_trits = {}
        self.last_raw_features = None

    def _scalar_sign(self, value: float, prev: int, hi: float, lo: float) -> int:
        trit = hysteretic_sign_trit(np.array([value], dtype=np.float32), hi=hi, lo=lo, previous=np.array([prev], dtype=np.int8))
        return int(trit[0])

    def compute_action(
        self,
        theta_field: np.ndarray,
        theta_dot_field: np.ndarray,
        env_state: np.ndarray,
        feature_vector: FeatureVector | None = None,
    ) -> Tuple[float, Dict[str, float]]:
        if feature_vector is None:
            feature_vector = self.extractor.extract(
                theta_field,
                theta_dot_field,
                env_state,
                prev_action=self.prev_action,
            )
        raw = feature_vector.raw
        theta_trit = self._scalar_sign(raw[4], self._prev_trits["theta"], self.config.theta_hi, self.config.theta_lo)
        theta_dot_trit = self._scalar_sign(raw[5], self._prev_trits["thetadot"], self.config.theta_dot_hi, self.config.theta_dot_lo)
        energy_trit = self._scalar_sign(raw[0], self._prev_trits["energy"], self.config.energy_hi, self.config.energy_lo)
        correction = ternary_majority(
            ternary_inv(theta_trit),
            ternary_inv(theta_dot_trit),
            ternary_inv(energy_trit),
        )
        direction = int(np.asarray(correction, dtype=np.int8).item())
        u_raw = direction * self.config.force_scale
        target = (1.0 - self.config.smooth_lambda) * self.prev_action + self.config.smooth_lambda * u_raw
        u = float(np.clip(target, -self.config.action_clip, self.config.action_clip))
        if self._in_deadzone(raw[4], raw[5]):
            direction = 0
            u = 0.0
        self.prev_action = u
        self._prev_trits["theta"] = theta_trit
        self._prev_trits["thetadot"] = theta_dot_trit
        self._prev_trits["energy"] = energy_trit
        self.last_trits = {"theta": theta_trit, "thetadot": theta_dot_trit, "energy": energy_trit, "direction": direction}
        self.last_raw_features = raw.copy()
        info = {
            "direction_trit": direction,
            "u_raw": u_raw,
            "u": u,
        }
        return u, info

    def _in_deadzone(self, theta: float, theta_dot: float) -> bool:
        return abs(theta) < self.config.deadzone_angle and abs(theta_dot) < self.config.deadzone_ang_vel
