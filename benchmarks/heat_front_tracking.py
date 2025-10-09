"""Heat diffusion benchmark for curvature-bounded front propagation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionParams,
)
from engine import total_energy_density
from engine.ops import laplacian

Array = np.ndarray


def generate_ring(shape: Tuple[int, int], radius: float, width: float) -> Array:
    h, w = shape
    yy, xx = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing="ij")
    dist = np.sqrt(xx ** 2 + yy ** 2)
    ring = np.exp(-((dist - radius) ** 2) / max(1e-6, width))
    return np.clip(ring.astype(np.float32), 0.0, 1.0)


@dataclass
class HeatFrontParams(HeatDiffusionParams):
    front_radius: float = 0.25
    front_width: float = 0.02


class HeatFrontTrackingSimulator:
    """Controller attempts to maintain a circular front with bounded curvature."""

    def __init__(self, params: HeatFrontParams, controller: HeatDiffusionController) -> None:
        self.params = params
        self.controller = controller
        self.target = generate_ring(params.shape, params.front_radius, params.front_width)
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        self.temp = generate_ring(self.params.shape, self.params.front_radius * 1.1, self.params.front_width)

    def step(self) -> Dict[str, float]:
        alpha, dt = self.params.alpha, self.params.dt
        lap = laplacian(self.temp)
        error = self.target - self.temp
        control = self.controller.step(error)
        self.temp += dt * (alpha * lap + control["delta"])

        mse = float(np.mean(error ** 2))
        energy = float(np.mean(total_energy_density(self.controller.theta, self.controller.theta_dot, self.controller.sim_cfg.physics)))
        curvature_proxy = float(np.mean(np.abs(lap)))
        return {
            "mse": mse,
            "energy": energy,
            "curvature_proxy": curvature_proxy,
        }

    def run(self, steps: int, record_interval: int = 50) -> Dict[str, object]:
        metrics = []
        history = []
        for step in range(steps):
            stats = self.step()
            if (step % record_interval) == 0 or step == steps - 1:
                metrics.append({"step": step, **stats})
                history.append(self.temp.copy())
        return {
            "metrics": metrics,
            "history": np.stack(history, axis=0) if history else np.empty((0,) + self.params.shape),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        history_arr: Array = history.get("history", np.empty(0))  # type: ignore[assignment]
        raster_path = out_path / "heat_front_target_vs_actual.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        target_im = axes[0].imshow(self.target, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0].set_title("Target front")
        fig.colorbar(target_im, ax=axes[0], fraction=0.046, pad=0.04)

        current = history_arr[-1] if history_arr.size else self.temp
        cur_im = axes[1].imshow(current, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[1].set_title("Current")
        fig.colorbar(cur_im, ax=axes[1], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)

        np.savez_compressed(out_path / "heat_front_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "heat_front_history.npz"),
        }


__all__ = [
    "HeatFrontParams",
    "HeatFrontTrackingSimulator",
    "generate_ring",
]
