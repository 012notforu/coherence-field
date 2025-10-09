"""Wave-field benchmark targeting standing cavity modes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import total_energy_density, load_config, accel_theta
from engine.integrators import leapfrog_step
from engine.ops import laplacian

Array = np.ndarray


@dataclass
class WaveCavityParams:
    shape: Tuple[int, int] = (96, 96)
    wave_speed: float = 1.0
    dt: float = 0.05
    damping: float = 0.001
    mode_m: int = 2
    mode_n: int = 3
    init_seed: int = 0


@dataclass
class WaveCavityControlConfig:
    scfd_cfg_path: str = "cfg/defaults.yaml"
    encode_gain: float = 0.5
    encode_decay: float = 0.9
    control_gain: float = 0.02
    control_clip: float = 0.1
    smooth_lambda: float = 0.3
    theta_clip: float = 2.0


class WaveCavityController:
    def __init__(self, cfg: WaveCavityControlConfig, grid_shape: Tuple[int, int]) -> None:
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


class WaveCavitySimulator:
    def __init__(self, params: WaveCavityParams, controller: WaveCavityController, target: Array) -> None:
        self.params = params
        self.controller = controller
        self.target = target.astype(np.float32)
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
        error = self.target - self.field
        error[self.boundary_mask] = 0.0
        control = self.controller.step(error)
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
        mse = float(np.mean((self.field - self.target) ** 2))
        boundary_energy = float(np.mean(self.field[self.boundary_mask] ** 2))
        return {
            "mse": mse,
            "energy": energy,
            "boundary_energy": boundary_energy,
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
        raster_path = out_dir / "wave_cavity_target_vs_actual.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(self.target, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[0].set_title("Target standing mode")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(history_array[-1], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[1].set_title("Current field")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)
        np.savez_compressed(out_dir / "wave_cavity_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_dir / "wave_cavity_history.npz"),
        }


def standing_mode_target(shape: Tuple[int, int], mode_m: int, mode_n: int) -> Array:
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)
    target = np.sin(np.pi * mode_m * (yy + 1) / (h + 1)) * np.sin(np.pi * mode_n * (xx + 1) / (w + 1))
    return target.astype(np.float32)


__all__ = [
    "WaveCavityParams",
    "WaveCavityControlConfig",
    "WaveCavityController",
    "WaveCavitySimulator",
    "standing_mode_target",
]
