"""Heat diffusion inverse problem benchmark."""
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
)

Array = np.ndarray


@dataclass
class HeatInverseParams:
    shape: Tuple[int, int] = (96, 96)
    alpha: float = 0.18
    dt: float = 0.1
    forward_steps: int = 40
    init_seed: int = 0
    source_scale: float = 0.5
    observation_noise: float = 0.0
    control_budget: float = 6.0


class HeatInverseSimulator:
    def __init__(
        self,
        params: HeatInverseParams,
        controller: HeatDiffusionController,
        *,
        source_kind: str = "blobs",
    ) -> None:
        self.params = params
        self.controller = controller
        self.source_kind = source_kind
        self.rng = np.random.default_rng(params.init_seed)
        self._budget_usage = 0.0
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        self._budget_usage = 0.0
        self.true_source = synthetic_source_map(self.params.shape, kind=self.source_kind, rng=self.rng)
        self.estimate = np.zeros(self.params.shape, dtype=np.float32)
        self.observed = self._forward_diffuse(self.true_source, add_noise=True)

    def _forward_diffuse(self, source: Array, *, add_noise: bool = False) -> Array:
        temp = np.zeros(self.params.shape, dtype=np.float32)
        for _ in range(self.params.forward_steps):
            lap = laplacian(temp)
            temp += self.params.dt * (self.params.alpha * lap + source)
        if add_noise and self.params.observation_noise > 0.0:
            temp += self.params.observation_noise * self.rng.standard_normal(temp.shape).astype(np.float32)
        return temp

    def step(self) -> Dict[str, float]:
        predicted = self._forward_diffuse(self.estimate, add_noise=False)
        error = self.observed - predicted
        control = self.controller.step(error)
        delta = control["delta"]

        if self.params.control_budget > 0.0:
            increment = float(np.mean(np.abs(delta)))
            projected = self._budget_usage + increment
            if projected > self.params.control_budget:
                scale = max(0.0, self.params.control_budget - self._budget_usage)
                if increment > 1e-6:
                    delta *= scale / increment
                self._budget_usage = self.params.control_budget
            else:
                self._budget_usage = projected
        self.estimate = np.clip(self.estimate + delta, -1.0, 1.0)

        predicted_after = self._forward_diffuse(self.estimate, add_noise=False)
        obs_mse = float(np.mean((predicted_after - self.observed) ** 2))
        source_mse = float(np.mean((self.estimate - self.true_source) ** 2))
        budget_util = 0.0 if self.params.control_budget <= 0.0 else min(1.0, self._budget_usage / self.params.control_budget)
        energy = float(
            np.mean(
                total_energy_density(
                    self.controller.theta,
                    self.controller.theta_dot,
                    self.controller.sim_cfg.physics,
                )
            )
        )
        return {
            "obs_mse": obs_mse,
            "source_mse": source_mse,
            "budget_util": budget_util,
            "energy": energy,
        }

    def run(self, steps: int, record_interval: int = 20) -> Dict[str, object]:
        metrics = []
        history_est = []
        history_err = []
        for t in range(steps):
            stats = self.step()
            if (t % record_interval) == 0 or t == steps - 1:
                metrics.append({"step": t, **stats})
                history_est.append(self.estimate.copy())
                history_err.append(self.observed - self._forward_diffuse(self.estimate, add_noise=False))
        return {
            "metrics": metrics,
            "estimate_history": np.stack(history_est, axis=0) if history_est else np.empty((0,) + self.params.shape),
            "error_history": np.stack(history_err, axis=0) if history_err else np.empty((0,) + self.params.shape),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        est_hist: Array = history.get("estimate_history", np.empty(0))
        raster_path = out_path / "heat_inverse_sources.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(self.true_source, cmap="inferno", vmin=-1.0, vmax=1.0)
        axes[0].set_title("True source")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        if est_hist.size:
            im1 = axes[1].imshow(est_hist[-1], cmap="inferno", vmin=-1.0, vmax=1.0)
        else:
            im1 = axes[1].imshow(self.estimate, cmap="inferno", vmin=-1.0, vmax=1.0)
        axes[1].set_title("Estimate")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        residual = self.true_source - self.estimate
        im2 = axes[2].imshow(residual, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[2].set_title("Residual")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)
        np.savez_compressed(out_path / "heat_inverse_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "heat_inverse_history.npz"),
        }


def synthetic_source_map(
    shape: Tuple[int, int],
    *,
    kind: str = "blobs",
    rng: np.random.Generator | None = None,
) -> Array:
    rng = np.random.default_rng() if rng is None else rng
    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    field = np.zeros((h, w), dtype=np.float32)
    if kind == "blobs":
        centers = rng.uniform(0.1, 0.9, size=(3, 2))
        scales = rng.uniform(0.02, 0.05, size=3)
        signs = rng.choice([-1.0, 1.0], size=3)
        for (cy, cx), scale, sign in zip(centers, scales, signs, strict=False):
            field += sign * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / scale)
    elif kind == "striped":
        field = np.sin(6.0 * np.pi * xx) * np.cos(4.0 * np.pi * yy)
    else:
        field = rng.standard_normal((h, w)).astype(np.float32)
    max_abs = np.max(np.abs(field))
    if max_abs > 0.0:
        field = field / max_abs
    return field.astype(np.float32)


__all__ = [
    "HeatInverseParams",
    "HeatInverseSimulator",
    "synthetic_source_map",
    "HeatDiffusionControlConfig",
    "HeatDiffusionController",
]
