"""Wave-field shaping benchmark with SCFD boundary modulation."""
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
class WaveFieldParams:
    shape: Tuple[int, int] = (96, 96)
    wave_speed: float = 1.0
    dt: float = 0.05
    damping: float = 0.001
    init_seed: int = 0


@dataclass
class WaveFieldControlConfig:
    scfd_cfg_path: str = "cfg/defaults.yaml"
    encode_gain: float = 0.5
    encode_decay: float = 0.9
    control_gain: float = 0.02
    control_clip: float = 0.1
    smooth_lambda: float = 0.3
    theta_clip: float = 2.0


class WaveFieldController:
    def __init__(self, cfg: WaveFieldControlConfig, grid_shape: Tuple[int, int]) -> None:
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


class WaveFieldSimulator:
    def __init__(
        self,
        params: WaveFieldParams,
        controller: WaveFieldController,
        target: Array,
    ) -> None:
        self.params = params
        self.controller = controller
        self.target = target.astype(np.float32)
        self.rng = np.random.default_rng(params.init_seed)
        self.shape = params.shape
        self._build_masks()
        self.reset()

    def _build_masks(self) -> None:
        h, w = self.shape
        boundary = np.zeros((h, w), dtype=bool)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True
        self.boundary_mask = boundary
        self.interior_mask = ~boundary

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.shape
        self.field = np.zeros((h, w), dtype=np.float32)
        self.velocity = np.zeros((h, w), dtype=np.float32)
        self.field += 0.01 * self.rng.standard_normal((h, w)).astype(np.float32)

    def step(self) -> Dict[str, float]:
        c, dt, damping = self.params.wave_speed, self.params.dt, self.params.damping
        lap = laplacian(self.field)
        error = np.zeros_like(self.field)
        error[self.interior_mask] = self.target[self.interior_mask] - self.field[self.interior_mask]
        control = self.controller.step(error)
        delta = np.zeros_like(self.field)
        delta[self.boundary_mask] = control["delta"][self.boundary_mask]

        self.velocity += dt * (c ** 2 * lap - damping * self.velocity + delta)
        self.field += dt * self.velocity
        energy = float(
            np.mean(
                total_energy_density(
                    self.controller.theta,
                    self.controller.theta_dot,
                    self.controller.sim_cfg.physics,
                )
            )
        )
        mse = float(np.mean((self.field - self.target) ** 2))
        return {
            "mse": mse,
            "energy": energy,
        }

    def run(self, steps: int, record_interval: int = 40) -> Dict[str, object]:
        metrics = []
        history = []
        for t in range(steps):
            stats = self.step()
            if (t % record_interval) == 0 or t == steps - 1:
                metrics.append({"step": t, **stats})
                history.append(self.field.copy())
        return {
            "metrics": metrics,
            "history": np.stack(history, axis=0),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        history_array: Array = history["history"]
        raster_path = out_path / "wave_field_target_vs_actual.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(self.target, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[0].set_title("Target Field")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(history_array[-1], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[1].set_title("Current Field")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)
        np.savez_compressed(out_path / "wave_field_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "wave_field_history.npz"),
        }


def synthetic_wave_target(shape: Tuple[int, int], kind: str = "focus") -> Array:
    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    if kind == "focus":
        pattern = np.exp(-((xx - 0.7) ** 2 + (yy - 0.5) ** 2) / 0.01)
    elif kind == "defocus":
        pattern = -np.exp(-((xx - 0.7) ** 2 + (yy - 0.5) ** 2) / 0.01)
    else:
        pattern = np.sin(6.0 * np.pi * xx) * np.sin(6.0 * np.pi * yy)
    return pattern.astype(np.float32)


__all__ = [
    "WaveFieldParams",
    "WaveFieldControlConfig",
    "WaveFieldController",
    "WaveFieldSimulator",
    "synthetic_wave_target",
]
