"""Gray-Scott reaction-diffusion benchmark with SCFD modulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import accel_theta, load_config, total_energy_density
from engine.integrators import leapfrog_step
from engine.ops import laplacian

Array = np.ndarray


@dataclass
class GrayScottParams:
    shape: Tuple[int, int] = (96, 96)
    D_u: float = 0.16
    D_v: float = 0.08
    F: float = 0.035
    k: float = 0.065
    dt: float = 1.0
    init_seed: int = 0
    noise: float = 0.02


@dataclass
class GrayScottControlConfig:
    scfd_cfg_path: str = "cfg/defaults.yaml"
    encode_gain: float = 0.5
    encode_decay: float = 0.9
    control_gain_feed: float = 0.002
    control_gain_kill: float = 0.002
    control_clip: float = 0.01
    smooth_lambda: float = 0.25
    theta_clip: float = 2.0


class GrayScottController:
    def __init__(self, params: GrayScottControlConfig, grid_shape: Tuple[int, int]) -> None:
        self.cfg = params
        self.sim_cfg = load_config(params.scfd_cfg_path)
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
            self.cfg.encode_decay * self.obs_filter + (1.0 - self.cfg.encode_decay) * error
        )
        inject = self.cfg.encode_gain * self._smooth(self.obs_filter)
        inject = np.clip(inject, -self.cfg.theta_clip, self.cfg.theta_clip)
        self.theta += inject
        self.theta = np.clip(self.theta, -self.cfg.theta_clip, self.cfg.theta_clip)

        self.theta, self.theta_dot, _, _ = leapfrog_step(
            self.theta,
            self.theta_dot,
            lambda f: accel_theta(f, self.sim_cfg.physics, dx=self.dx),
            self.sim_cfg.integration.dt,
            max_step=None,
        )
        control = np.tanh(self.theta)
        feed_delta = np.clip(self.cfg.control_gain_feed * control, -self.cfg.control_clip, self.cfg.control_clip)
        kill_delta = np.clip(self.cfg.control_gain_kill * control, -self.cfg.control_clip, self.cfg.control_clip)
        return {
            "feed_delta": feed_delta,
            "kill_delta": kill_delta,
            "theta": self.theta.copy(),
        }


class GrayScottSimulator:
    def __init__(
        self,
        params: GrayScottParams,
        controller: GrayScottController,
        target_pattern: Array,
    ) -> None:
        self.params = params
        self.controller = controller
        self.target = target_pattern.astype(np.float32)
        self.rng = np.random.default_rng(params.init_seed)
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.params.shape
        self.u = np.ones((h, w), dtype=np.float32)
        self.v = np.zeros((h, w), dtype=np.float32)
        # Seed square of V in the center with noise
        pad = h // 10
        self.v[pad:-pad, pad:-pad] = 0.5 + 0.1 * self.rng.standard_normal((h - 2 * pad, w - 2 * pad))
        self.u -= self.v
        self.u += self.params.noise * self.rng.standard_normal(self.u.shape)
        self.v += self.params.noise * self.rng.standard_normal(self.v.shape)

    def step(self) -> Dict[str, float]:
        D_u, D_v, F, k, dt = self.params.D_u, self.params.D_v, self.params.F, self.params.k, self.params.dt
        lap_u = laplacian(self.u)
        lap_v = laplacian(self.v)
        uvv = self.u * (self.v ** 2)
        error = self.target - self.v
        control = self.controller.step(error)
        F_eff = np.clip(F + control["feed_delta"], 0.0, 0.08)
        k_eff = np.clip(k + control["kill_delta"], 0.0, 0.1)
        du = D_u * lap_u - uvv + F_eff * (1.0 - self.u)
        dv = D_v * lap_v + uvv - (F_eff + k_eff) * self.v
        self.u += dt * du
        self.v += dt * dv
        mse = float(np.mean((self.v - self.target) ** 2))
        energy = float(np.mean(total_energy_density(self.controller.theta, self.controller.theta_dot, self.controller.sim_cfg.physics)))
        return {
            "mse": mse,
            "energy": energy,
        }

    def run(self, steps: int, record_interval: int = 50) -> Dict[str, object]:
        metrics = []
        history_u = []
        history_v = []
        for t in range(steps):
            stats = self.step()
            if (t % record_interval) == 0 or t == steps - 1:
                metrics.append({"step": t, **stats})
                history_u.append(self.u.copy())
                history_v.append(self.v.copy())
        return {
            "metrics": metrics,
            "u_history": np.stack(history_u, axis=0),
            "v_history": np.stack(history_v, axis=0),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        v_history: Array = history["v_history"]
        raster = v_history[-1]
        raster_path = out_path / "gray_scott_target_vs_actual.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(self.target, cmap="magma", vmin=0.0, vmax=1.0)
        axes[0].set_title("Target V")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(raster, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1].set_title("Current V")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)
        np.savez_compressed(out_path / "gray_scott_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "gray_scott_history.npz"),
        }


def synthetic_target(shape: Tuple[int, int], kind: str = "spots") -> Array:
    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    if kind == "stripes":
        pattern = 0.5 + 0.5 * np.sin(10.0 * xx)
    elif kind == "spots":
        pattern = 0.5 + 0.5 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)
        pattern += 0.2 * np.exp(-((xx - 0.25) ** 2 + (yy - 0.25) ** 2) / 0.01)
        pattern += 0.2 * np.exp(-((xx - 0.75) ** 2 + (yy - 0.75) ** 2) / 0.01)
    elif kind == "hover":
        pattern = 0.5
        pattern += 0.12 * (np.sin(4.0 * np.pi * xx) + np.sin(4.0 * np.pi * yy))
        pattern += 0.06 * np.sin(6.0 * np.pi * (xx + yy))
        pattern += 0.04 * np.sin(8.0 * np.pi * (xx - yy))
    elif kind == "checker":
        pattern = 0.5 + 0.5 * (np.sign(np.sin(8.0 * np.pi * xx) * np.sin(8.0 * np.pi * yy)))
    else:
        pattern = 0.5 + 0.5 * np.sin(8.0 * xx) * np.sin(8.0 * yy)
    return np.clip(pattern.astype(np.float32), 0.0, 1.0)


__all__ = [
    "GrayScottParams",
    "GrayScottControlConfig",
    "GrayScottController",
    "GrayScottSimulator",
    "synthetic_target",
]
