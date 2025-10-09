"""Heat diffusion benchmark with a mobile actuator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import accel_theta, load_config, total_energy_density
from engine.integrators import leapfrog_step
from engine.ops import laplacian

from .heat_diffusion import (
    HeatDiffusionParams,
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    synthetic_temperature,
)

Array = np.ndarray


@dataclass
class HeatMobileParams(HeatDiffusionParams):
    """Heat diffusion parameters extended with a scripted actuator path."""

    path: Tuple[Tuple[int, int], ...] = ((48, 16), (48, 48), (48, 80))
    steps_per_waypoint: int = 40
    heater_radius: int = 3
    heater_amplitude: float = 0.6
    control_budget: float = 12.0


class HeatMobileSimulator:
    """Simulator that injects heat along a moving actuator path."""

    def __init__(
        self,
        params: HeatMobileParams,
        controller: HeatDiffusionController,
        target_kind: str = "moving_gaussian",
    ) -> None:
        self.params = params
        self.controller = controller
        self.target_kind = target_kind
        self.rng = np.random.default_rng(params.init_seed)
        self._steps = 0
        self._budget_usage = 0.0
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.params.shape
        self.temp = 0.25 * np.ones((h, w), dtype=np.float32)
        self.temp += self.params.noise * self.rng.standard_normal((h, w)).astype(np.float32)
        self._steps = 0
        self._budget_usage = 0.0

    # ------------------------------------------------------------------
    def _current_waypoint(self) -> Tuple[int, int]:
        idx = min(
            self._steps // max(1, self.params.steps_per_waypoint),
            len(self.params.path) - 1,
        )
        return self.params.path[idx]

    def _heater_mask(self) -> Array:
        h, w = self.params.shape
        mask = np.zeros((h, w), dtype=np.float32)
        yc, xc = self._current_waypoint()
        radius = max(1, self.params.heater_radius)
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        dist_sq = (y - yc) ** 2 + (x - xc) ** 2
        mask[dist_sq <= radius**2] = 1.0
        if mask.sum() > 0:
            mask /= float(mask.sum())
        return mask

    def _target_field(self) -> Array:
        if self.target_kind == "moving_gaussian":
            center = self._current_waypoint()
            return synthetic_temperature(self.params.shape, kind="hot_corner", center=center)
        return synthetic_temperature(self.params.shape, kind="gradient")

    # ------------------------------------------------------------------
    def step(self) -> Dict[str, float]:
        alpha, dt = self.params.alpha, self.params.dt
        lap = laplacian(self.temp)
        target = self._target_field()
        error = target - self.temp
        control = self.controller.step(error)

        heater = self.params.heater_amplitude * self._heater_mask()
        self.temp += dt * (alpha * lap + control["delta"] + heater)

        self._budget_usage += float(np.mean(np.abs(control["delta"])))
        self._steps += 1

        mse = float(np.mean(error**2))
        energy = float(
            np.mean(
                total_energy_density(
                    self.controller.theta,
                    self.controller.theta_dot,
                    self.controller.sim_cfg.physics,
                )
            )
        )
        path_progress = float(self._steps / (len(self.params.path) * max(1, self.params.steps_per_waypoint)))
        budget = self.params.control_budget or 1.0
        budget_util = float(min(1.0, self._budget_usage / budget))

        return {
            "mse": mse,
            "energy": energy,
            "budget_util": budget_util,
            "path_progress": path_progress,
        }

    def run(self, steps: int, record_interval: int = 50) -> Dict[str, Array | List[Dict[str, float]]]:
        metrics: List[Dict[str, float]] = []
        history = []
        for _ in range(steps):
            stats = self.step()
            if self._steps % record_interval == 0 or self._steps == steps:
                metrics.append({"step": self._steps, **stats})
                history.append(self.temp.copy())
        return {
            "metrics": metrics,
            "temps": np.stack(history, axis=0) if history else np.empty((0,) + self.params.shape),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        temps: Array = history["temps"] if "temps" in history else np.empty((0,))
        raster_path = out_path / "heat_mobile_latest.png"
        fig, ax = plt.subplots(figsize=(4, 4))
        if temps.size > 0:
            im = ax.imshow(temps[-1], cmap="inferno", vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Temperature")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)

        np.savez_compressed(out_path / "heat_mobile_history.npz", temps=temps)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "heat_mobile_history.npz"),
        }


def synthetic_mobile_target(shape: Tuple[int, int], path: Sequence[Tuple[int, int]], radius: int = 4) -> Array:
    """Generate a stack of targets aligned with the mobile path."""

    h, w = shape
    targets = []
    for yc, xc in path:
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        dist_sq = (yy - yc) ** 2 + (xx - xc) ** 2
        pattern = np.exp(-dist_sq / max(1.0, float(radius**2)))
        pattern = np.clip(pattern, 0.0, 1.0)
        pattern /= pattern.max() if pattern.max() > 0 else 1.0
        targets.append(pattern.astype(np.float32))
    return np.stack(targets, axis=0)


__all__ = [
    "HeatMobileParams",
    "HeatMobileSimulator",
    "synthetic_mobile_target",
    "HeatDiffusionControlConfig",
    "HeatDiffusionController",
]
