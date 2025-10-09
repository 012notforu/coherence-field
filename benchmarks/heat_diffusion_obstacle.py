"""Heat diffusion benchmark with interior obstacles and control budget."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import total_energy_density
from engine.ops import laplacian

from .heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionParams,
    HeatDiffusionSimulator,
    synthetic_temperature,
)

Array = np.ndarray


@dataclass
class HeatObstacleParams(HeatDiffusionParams):
    obstacle_temperature: float = 0.05
    control_budget: float = 6.0
    gap_height: int = 6
    gap_width: int = 4


class HeatObstacleSimulator(HeatDiffusionSimulator):
    """Heat simulator with a central obstacle and per-step control budget."""

    def __init__(
        self,
        params: HeatObstacleParams,
        controller: HeatDiffusionController,
        target: Array,
    ) -> None:
        self.params = params
        self.obstacle_mask = self._build_obstacle_mask(params.shape, params.gap_height, params.gap_width)
        self.corner_mask = self._corner_mask(params.shape)
        self.prev_energy: float | None = None
        super().__init__(params, controller, target)

    @staticmethod
    def _build_obstacle_mask(shape: Tuple[int, int], gap_h: int, gap_w: int) -> Array:
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        top = h // 3
        bottom = 2 * h // 3
        left = w // 3
        right = 2 * w // 3
        mask[top:bottom, left:right] = True
        gap_top = max(top, top + gap_h)
        mask[top:gap_top, left:right] = False
        center = (left + right) // 2
        mask[gap_top:bottom, center - gap_w:center + gap_w] = False
        return mask

    @staticmethod
    def _corner_mask(shape: Tuple[int, int]) -> Array:
        h, w = shape
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 4, : w // 4] = True
        return mask

    def reset(self) -> None:
        super().reset()
        self.prev_energy = None
        self.temp[self.obstacle_mask] = self.params.obstacle_temperature

    def _apply_budget(self, delta: Array) -> tuple[Array, float, float]:
        total = float(np.sum(np.abs(delta)))
        budget = self.params.control_budget
        if budget is None or budget <= 0.0 or total <= budget:
            util = total / max(budget, 1e-8) if budget else 1.0
            return delta, total, util
        scale = budget / (total + 1e-8)
        return delta * scale, total, min(1.0, scale * total / max(budget, 1e-8))

    def step(self) -> Dict[str, float]:
        alpha = self.params.alpha
        dt = self.params.dt
        lap = laplacian(self.temp)
        lap[self.obstacle_mask] = 0.0

        error = self.target - self.temp
        error[self.obstacle_mask] = 0.0
        control = self.controller.step(error)
        raw_delta = control["delta"]
        raw_delta[self.obstacle_mask] = 0.0
        delta, total_control, budget_util = self._apply_budget(raw_delta)
        self.temp += dt * (alpha * lap + delta)
        self.temp[self.obstacle_mask] = self.params.obstacle_temperature

        energy = float(
            np.mean(
                total_energy_density(
                    self.controller.theta,
                    self.controller.theta_dot,
                    self.controller.sim_cfg.physics,
                )
            )
        )
        delta_energy = 0.0 if self.prev_energy is None else energy - self.prev_energy
        self.prev_energy = energy

        mse = float(np.mean((self.target - self.temp) ** 2))
        corner_err = float(np.mean((self.target[self.corner_mask] - self.temp[self.corner_mask]) ** 2))
        control_norm = float(np.linalg.norm(delta))

        return {
            "mse": mse,
            "corner_mse": corner_err,
            "energy": energy,
            "delta_energy": delta_energy,
            "control_total_l1": total_control,
            "budget_utilisation": budget_util,
            "control_norm": control_norm,
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        artifacts = super().generate_visualization(out_dir, history)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        mask_path = out_path / "heat_obstacle_mask.png"
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(self.obstacle_mask, cmap="gray_r")
        ax.set_title("Obstacle mask")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(mask_path, dpi=160)
        plt.close(fig)

        artifacts["obstacle_mask"] = str(mask_path)
        return artifacts


def synthetic_obstacle_target(
    shape: Tuple[int, int],
    kind: str = "hot_corner",
) -> Array:
    if kind == "hot_corner":
        base = synthetic_temperature(shape, kind="hot_corner")
    elif kind == "cool_corner":
        base = 1.0 - synthetic_temperature(shape, kind="hot_corner")
    else:
        base = synthetic_temperature(shape, kind="gradient")
    return base.astype(np.float32)


__all__ = [
    "HeatObstacleParams",
    "HeatObstacleSimulator",
    "synthetic_obstacle_target",
    "HeatDiffusionControlConfig",
    "HeatDiffusionController",
]
