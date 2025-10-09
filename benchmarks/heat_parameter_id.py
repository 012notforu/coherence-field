"""Heat diffusion parameter identification benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.heat_diffusion import HeatDiffusionControlConfig, HeatDiffusionController, HeatDiffusionParams, synthetic_temperature
from engine import total_energy_density
from engine.ops import laplacian

Array = np.ndarray


@dataclass
class HeatParameterIDParams(HeatDiffusionParams):
    alpha_low: float = 0.12
    alpha_high: float = 0.28
    split_axis: str = "vertical"  # "vertical" or "horizontal"


class HeatParameterIDSimulator:
    """Controller must estimate hidden diffusivity map while tracking temperature."""

    def __init__(self, params: HeatParameterIDParams, controller: HeatDiffusionController) -> None:
        self.params = params
        self.controller = controller
        self.hidden_alpha = self._build_alpha_map()
        self.alpha_estimate = np.full(params.shape, params.alpha, dtype=np.float32)
        self.target = synthetic_temperature(params.shape, kind="gradient")
        self.reset()

    def _build_alpha_map(self) -> Array:
        h, w = self.params.shape
        alpha_map = np.full((h, w), self.params.alpha_low, dtype=np.float32)
        if self.params.split_axis == "vertical":
            alpha_map[:, w // 2 :] = self.params.alpha_high
        else:
            alpha_map[h // 2 :, :] = self.params.alpha_high
        return alpha_map

    def reset(self) -> None:
        self.controller.reset()
        self.temp = 0.25 * np.ones(self.params.shape, dtype=np.float32)
        self.temp += self.params.noise * self.controller.sim_cfg.integration.dt * self.controller.sim_cfg.integration.dt
        self.alpha_estimate.fill(self.params.alpha)

    def step(self) -> Dict[str, float]:
        dt = self.params.dt
        lap = laplacian(self.temp)
        error = self.target - self.temp
        control = self.controller.step(error)
        self.alpha_estimate = np.clip(self.alpha_estimate + 0.001 * control["delta"], self.params.alpha_low * 0.5, self.params.alpha_high * 1.5)
        self.temp += dt * (self.hidden_alpha * lap + control["delta"])

        mse = float(np.mean(error ** 2))
        energy = float(np.mean(total_energy_density(self.controller.theta, self.controller.theta_dot, self.controller.sim_cfg.physics)))
        alpha_rmse = float(np.sqrt(np.mean((self.alpha_estimate - self.hidden_alpha) ** 2)))
        return {
            "mse": mse,
            "energy": energy,
            "alpha_rmse": alpha_rmse,
        }

    def run(self, steps: int, record_interval: int = 50) -> Dict[str, object]:
        metrics = []
        history_temp = []
        history_alpha = []
        for step in range(steps):
            stats = self.step()
            if (step % record_interval) == 0 or step == steps - 1:
                metrics.append({"step": step, **stats})
                history_temp.append(self.temp.copy())
                history_alpha.append(self.alpha_estimate.copy())
        return {
            "metrics": metrics,
            "temperature_history": np.stack(history_temp, axis=0) if history_temp else np.empty((0,) + self.params.shape),
            "alpha_history": np.stack(history_alpha, axis=0) if history_alpha else np.empty((0,) + self.params.shape),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        alpha_hist: Array = history.get("alpha_history", np.empty(0))  # type: ignore[assignment]
        temp_hist: Array = history.get("temperature_history", np.empty(0))  # type: ignore[assignment]

        raster_path = out_path / "heat_param_id_alpha.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(self.hidden_alpha, cmap="viridis", vmin=self.params.alpha_low, vmax=self.params.alpha_high)
        axes[0].set_title("True alpha map")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        estimate = alpha_hist[-1] if alpha_hist.size else self.alpha_estimate
        im1 = axes[1].imshow(estimate, cmap="viridis", vmin=self.params.alpha_low, vmax=self.params.alpha_high)
        axes[1].set_title("Estimated alpha")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        current = temp_hist[-1] if temp_hist.size else self.temp
        im2 = axes[2].imshow(current, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title("Temperature")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)

        np.savez_compressed(out_path / "heat_param_id_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "heat_param_id_history.npz"),
        }


__all__ = [
    "HeatParameterIDParams",
    "HeatParameterIDSimulator",
]
