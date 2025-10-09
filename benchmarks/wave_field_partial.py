"""Wave-field benchmark with partial sensors and action delay."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import total_energy_density, load_config, accel_theta
from engine.integrators import leapfrog_step
from engine.ops import laplacian

Array = np.ndarray


@dataclass
class WavePartialParams:
    shape: Tuple[int, int] = (96, 96)
    wave_speed: float = 1.0
    dt: float = 0.05
    damping: float = 0.001
    init_seed: int = 0
    sensor_fraction: float = 0.25
    action_delay: int = 3


@dataclass
class WavePartialControlConfig:
    scfd_cfg_path: str = "cfg/defaults.yaml"
    encode_gain: float = 0.5
    encode_decay: float = 0.9
    control_gain: float = 0.02
    control_clip: float = 0.1
    smooth_lambda: float = 0.3
    theta_clip: float = 2.0


class WavePartialController:
    def __init__(self, cfg: WavePartialControlConfig, grid_shape: Tuple[int, int], action_delay: int) -> None:
        self.cfg = cfg
        self.sim_cfg = load_config(cfg.scfd_cfg_path)
        self.dx = self.sim_cfg.grid.spacing
        self.theta = np.zeros(grid_shape, dtype=np.float32)
        self.theta_dot = np.zeros_like(self.theta)
        self.obs_filter = np.zeros(grid_shape, dtype=np.float32)
        self.delay = max(0, action_delay)
        self.buffer: Deque[Array] = deque(maxlen=self.delay + 1)
        for _ in range(self.delay + 1):
            self.buffer.append(np.zeros(grid_shape, dtype=np.float32))

    def reset(self) -> None:
        self.theta.fill(0.0)
        self.theta_dot.fill(0.0)
        self.obs_filter.fill(0.0)
        self.buffer.clear()
        for _ in range(self.delay + 1):
            self.buffer.append(np.zeros_like(self.theta))

    def _smooth(self, field: Array) -> Array:
        acc = field.copy()
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += np.roll(field, shift=shift, axis=(0, 1))
        return acc / 5.0

    def step(self, sensed_error: Array) -> Dict[str, Array]:
        error = sensed_error.astype(np.float32)
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
        self.buffer.append(delta)
        applied = self.buffer[0]
        return {
            "delta": applied,
            "queued": delta,
        }


class WavePartialSimulator:
    def __init__(
        self,
        params: WavePartialParams,
        controller: WavePartialController,
        target: Array,
        sensor_mask: Array,
    ) -> None:
        self.params = params
        self.controller = controller
        self.target = target.astype(np.float32)
        self.sensor_mask = sensor_mask.astype(np.float32)
        self.rng = np.random.default_rng(params.init_seed)
        self.shape = params.shape
        self.boundary_mask = self._build_boundary_mask()
        self.reset()

    def _build_boundary_mask(self) -> Array:
        mask = np.zeros(self.shape, dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return mask

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.shape
        self.field = np.zeros((h, w), dtype=np.float32)
        self.velocity = np.zeros((h, w), dtype=np.float32)
        self.field += 0.01 * self.rng.standard_normal((h, w)).astype(np.float32)

    def _apply_boundaries(self) -> None:
        self.field[self.boundary_mask] = 0.0
        self.velocity[self.boundary_mask] = 0.0

    def step(self) -> Dict[str, float]:
        c, dt, damping = self.params.wave_speed, self.params.dt, self.params.damping
        lap = laplacian(self.field)
        full_error = self.target - self.field
        sensed_error = full_error * self.sensor_mask
        control = self.controller.step(sensed_error)
        delta = control["delta"]
        self.velocity += dt * (c ** 2 * lap - damping * self.velocity + delta)
        self.field += dt * self.velocity
        self._apply_boundaries()

        energy = float(
            np.mean(
                total_energy_density(
                    self.controller.theta,
                    self.controller.theta_dot,
                    self.controller.sim_cfg.physics,
                )
            )
        )
        mse = float(np.mean(full_error ** 2))
        sensed_mse = float(np.mean((full_error * self.sensor_mask) ** 2))
        control_norm = float(np.linalg.norm(delta))
        coverage = float(np.mean(self.sensor_mask))
        return {
            "mse": mse,
            "sensed_mse": sensed_mse,
            "energy": energy,
            "control_norm": control_norm,
            "sensor_coverage": coverage,
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

    def generate_visualization(self, out_dir: Path, history: Dict[str, object]) -> Dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        history_array: Array = history["history"]
        raster_path = out_dir / "wave_partial_target_vs_actual.png"
        mask_path = out_dir / "wave_partial_sensor_mask.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(self.target, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[0].set_title("Target field")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(history_array[-1], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[1].set_title("Current field")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.imshow(self.sensor_mask, cmap="gray_r")
        ax2.set_title("Sensor mask")
        ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(mask_path, dpi=160)
        plt.close(fig2)
        np.savez_compressed(out_dir / "wave_partial_history.npz", **history)
        return {
            "raster": str(raster_path),
            "sensor_mask": str(mask_path),
            "history": str(out_dir / "wave_partial_history.npz"),
        }


def random_sensor_mask(shape: Tuple[int, int], fraction: float, rng: np.random.Generator) -> Array:
    h, w = shape
    total = h * w
    k = int(total * np.clip(fraction, 0.05, 1.0))
    mask = np.zeros(total, dtype=np.float32)
    indices = rng.choice(total, size=k, replace=False)
    mask[indices] = 1.0
    return mask.reshape(shape)


__all__ = [
    "WavePartialParams",
    "WavePartialControlConfig",
    "WavePartialController",
    "WavePartialSimulator",
    "random_sensor_mask",
]
