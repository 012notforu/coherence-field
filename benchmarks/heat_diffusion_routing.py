"""Heat diffusion routing benchmark: move multiple blobs without collision."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionParams,
    synthetic_temperature,
)
from engine import total_energy_density
from engine.ops import laplacian

Array = np.ndarray


@dataclass
class HeatDiffusionRoutingParams(HeatDiffusionParams):
    initial_centers: Sequence[Tuple[float, float]] = ((0.3, 0.3), (0.7, 0.7))
    target_centers: Sequence[Tuple[float, float]] = ((0.7, 0.3), (0.3, 0.7))
    blob_sigma: float = 0.02
    collision_radius: float = 0.08


def generate_blob_pattern(
    shape: Tuple[int, int],
    centers: Iterable[Tuple[float, float]],
    sigma: float,
) -> Array:
    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    pattern = np.zeros((h, w), dtype=np.float32)
    inv_sigma = 1.0 / max(1e-6, sigma)
    for cy, cx in centers:
        pattern += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) * inv_sigma)
    pattern = np.clip(pattern, 0.0, 1.0)
    return pattern.astype(np.float32)


class HeatDiffusionRoutingSimulator:
    """Controller tasked with routing multiple blobs to target positions."""

    def __init__(
        self,
        params: HeatDiffusionRoutingParams,
        controller: HeatDiffusionController,
    ) -> None:
        self.params = params
        self.controller = controller
        self.rng = np.random.default_rng(params.init_seed)
        self.target = generate_blob_pattern(params.shape, params.target_centers, params.blob_sigma)
        self.collision_mask = self._build_collision_mask()
        self.reset()

    def _build_collision_mask(self) -> Array:
        h, w = self.params.shape
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
        centers = np.array(self.params.initial_centers, dtype=np.float32)
        centroid = centers.mean(axis=0)
        dist = np.sqrt((yy - centroid[0]) ** 2 + (xx - centroid[1]) ** 2)
        return (dist <= self.params.collision_radius).astype(np.float32)

    def reset(self) -> None:
        self.controller.reset()
        self.temp = generate_blob_pattern(self.params.shape, self.params.initial_centers, self.params.blob_sigma)
        self.temp += self.params.noise * self.rng.standard_normal(self.temp.shape).astype(np.float32)

    def step(self) -> Dict[str, float]:
        alpha, dt = self.params.alpha, self.params.dt
        lap = laplacian(self.temp)
        error = self.target - self.temp
        control = self.controller.step(error)
        self.temp += dt * (alpha * lap + control["delta"])

        mse = float(np.mean(error ** 2))
        energy = float(np.mean(total_energy_density(self.controller.theta, self.controller.theta_dot, self.controller.sim_cfg.physics)))
        collision_penalty = float(np.mean(self.temp * self.collision_mask))
        return {
            "mse": mse,
            "energy": energy,
            "collision_penalty": collision_penalty,
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
        raster_path = out_path / "heat_routing_target_vs_actual.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        target_im = axes[0].imshow(self.target, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0].set_title("Target blobs")
        fig.colorbar(target_im, ax=axes[0], fraction=0.046, pad=0.04)

        current = history_arr[-1] if history_arr.size else self.temp
        current_im = axes[1].imshow(current, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[1].set_title("Current")
        fig.colorbar(current_im, ax=axes[1], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)

        np.savez_compressed(out_path / "heat_routing_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "heat_routing_history.npz"),
        }


__all__ = [
    "HeatDiffusionRoutingParams",
    "HeatDiffusionRoutingSimulator",
    "generate_blob_pattern",
]
