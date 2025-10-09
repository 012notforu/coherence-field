"""Heat diffusion benchmark with SCFD boundary modulation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import accel_theta, load_config, total_energy_density
from engine.integrators import leapfrog_step
from engine.ops import laplacian

Array = np.ndarray


@dataclass
class HeatDiffusionParams:
    shape: Tuple[int, int] = (96, 96)
    alpha: float = 0.18
    dt: float = 0.1
    init_seed: int = 0
    noise: float = 0.01


@dataclass
class HeatDiffusionControlConfig:
    scfd_cfg_path: str = "cfg/defaults.yaml"
    encode_gain: float = 0.4
    encode_decay: float = 0.95
    control_gain: float = 0.005
    control_clip: float = 0.05
    smooth_lambda: float = 0.3
    theta_clip: float = 2.0


class HeatDiffusionController:
    def __init__(self, cfg: HeatDiffusionControlConfig, grid_shape: Tuple[int, int]) -> None:
        self.cfg = cfg
        self.sim_cfg = load_config(cfg.scfd_cfg_path)
        self.dx = self.sim_cfg.grid.spacing
        self.theta = np.zeros(grid_shape, dtype=np.float32)
        self.theta_dot = np.zeros_like(self.theta)
        self.obs_filter = np.zeros(grid_shape, dtype=np.float32)

    def reset(self) -> None:
        self.theta.fill(0.0)
        self.theta_dot.fill(0.0)
        self.obs_filter.fill(0.0)

    def _smooth(self, field: Array) -> Array:
        acc = field.copy()
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(field, shift=shift, axis=(0, 1))
        return acc / 5.0

    def step(self, error_field: Array) -> Dict[str, Array]:
        error = error_field.astype(np.float32)
        self.obs_filter = (
            self.cfg.encode_decay * self.obs_filter
            + (1.0 - self.cfg.encode_decay) * error
        )
        inject = self.cfg.encode_gain * self._smooth(self.obs_filter)
        target_theta = np.clip(self.theta + inject, -self.cfg.theta_clip, self.cfg.theta_clip)
        self.theta = (
            (1.0 - self.cfg.smooth_lambda) * self.theta
            + self.cfg.smooth_lambda * target_theta
        )

        self.theta, self.theta_dot, _, _ = leapfrog_step(
            self.theta,
            self.theta_dot,
            lambda f: accel_theta(f, self.sim_cfg.physics, dx=self.dx),
            self.sim_cfg.integration.dt,
            max_step=None,
        )
        control = np.tanh(self.theta)
        delta = np.clip(self.cfg.control_gain * control, -self.cfg.control_clip, self.cfg.control_clip)
        return {
            "delta": delta,
            "theta": self.theta.copy(),
        }


class HeatDiffusionSimulator:
    def __init__(
        self,
        params: HeatDiffusionParams,
        controller: HeatDiffusionController,
        target: Array,
    ) -> None:
        self.params = params
        self.controller = controller
        self.target = target.astype(np.float32)
        self.rng = np.random.default_rng(params.init_seed)
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.params.shape
        self.temp = 0.25 * np.ones((h, w), dtype=np.float32)
        self.temp += self.params.noise * self.rng.standard_normal((h, w)).astype(np.float32)

    def step(self) -> Dict[str, float]:
        alpha, dt = self.params.alpha, self.params.dt
        lap = laplacian(self.temp)
        error = self.target - self.temp
        control = self.controller.step(error)
        self.temp += dt * (alpha * lap + control["delta"])
        mse = float(np.mean(error ** 2))
        energy = float(np.mean(total_energy_density(self.controller.theta, self.controller.theta_dot, self.controller.sim_cfg.physics)))
        mean_abs_grad = float(np.mean(np.abs(np.gradient(self.temp)[0])) + np.mean(np.abs(np.gradient(self.temp)[1])))
        return {
            "mse": mse,
            "energy": energy,
            "mean_abs_grad": mean_abs_grad,
        }

    def run(self, steps: int, record_interval: int = 50) -> Dict[str, object]:
        metrics = []
        history = []
        for t in range(steps):
            stats = self.step()
            if (t % record_interval) == 0 or t == steps - 1:
                metrics.append({"step": t, **stats})
                history.append(self.temp.copy())
        return {
            "metrics": metrics,
            "history": np.stack(history, axis=0),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        history_array: Array = history["history"]
        raster_path = out_path / "heat_diffusion_target_vs_actual.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(self.target, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0].set_title("Target Temperature")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(history_array[-1], cmap="inferno", vmin=0.0, vmax=1.0)
        axes[1].set_title("Current Temperature")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)
        np.savez_compressed(out_path / "heat_diffusion_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "heat_diffusion_history.npz"),
        }


def synthetic_temperature(
    shape: Tuple[int, int],
    kind: str = "gradient",
    center: Tuple[float, float] | Tuple[int, int] | None = None,
) -> Array:
    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")

    def _normalize(pair: Tuple[float, float] | Tuple[int, int]) -> tuple[float, float]:
        cy, cx = pair
        cy = float(cy)
        cx = float(cx)
        if cy > 1.0 or cx > 1.0:
            cy /= max(1.0, float(h - 1))
            cx /= max(1.0, float(w - 1))
        return cy, cx

    if kind == "gradient":
        pattern = xx
    elif kind == "hot_corner":
        cy, cx = _normalize(center) if center is not None else (0.2, 0.2)
        pattern = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.02)
    elif kind == "cool_spot":
        cy, cx = _normalize(center) if center is not None else (0.8, 0.8)
        pattern = 1.0 - 0.8 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.02)
    else:
        if center is not None:
            cy, cx = _normalize(center)
            phase = 2.0 * np.pi * (xx * cx + yy * cy)
            pattern = 0.5 + 0.3 * np.sin(phase)
        else:
            pattern = 0.5 + 0.3 * np.sin(4.0 * np.pi * xx) * np.sin(4.0 * np.pi * yy)
    return np.clip(pattern.astype(np.float32), 0.0, 1.0)


__all__ = [
    "HeatDiffusionParams",
    "HeatDiffusionControlConfig",
    "HeatDiffusionController",
    "HeatDiffusionSimulator",
    "synthetic_temperature",
]

