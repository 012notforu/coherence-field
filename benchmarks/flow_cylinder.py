"""Flow control benchmark with SCFD-driven cylinder jets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import accel_theta, load_config, total_energy_density
from engine.integrators import leapfrog_step
from engine.ops import laplacian, grad

Array = np.ndarray


@dataclass
class FlowCylinderParams:
    shape: Tuple[int, int] = (96, 96)
    viscosity: float = 0.02
    dt: float = 0.05
    inflow: float = 1.0
    cylinder_radius: int = 8
    init_seed: int = 0


@dataclass
class FlowCylinderControlConfig:
    scfd_cfg_path: str = "cfg/defaults.yaml"
    encode_gain: float = 0.4
    encode_decay: float = 0.93
    control_gain: float = 0.02
    control_clip: float = 0.1
    smooth_lambda: float = 0.3
    theta_clip: float = 2.0


class FlowCylinderController:
    def __init__(self, cfg: FlowCylinderControlConfig, grid_shape: Tuple[int, int]) -> None:
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


class FlowCylinderSimulator:
    def __init__(
        self,
        params: FlowCylinderParams,
        controller: FlowCylinderController,
    ) -> None:
        self.params = params
        self.controller = controller
        self.rng = np.random.default_rng(params.init_seed)
        self.shape = params.shape
        self._build_masks()
        self.reset()

    def _build_masks(self) -> None:
        h, w = self.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx, cy = h // 2, w // 3
        dist = np.sqrt((yy - cx) ** 2 + (xx - cy) ** 2)
        self.cylinder_mask = dist <= self.params.cylinder_radius
        self.jet_mask = np.zeros_like(self.cylinder_mask, dtype=np.float32)
        jet_band = (dist >= self.params.cylinder_radius - 1) & (dist <= self.params.cylinder_radius + 1)
        left_sector = (xx < cy) & (np.abs(yy - cx) <= self.params.cylinder_radius // 2)
        right_sector = (xx > cy) & (np.abs(yy - cx) <= self.params.cylinder_radius // 2)
        self.jet_mask[left_sector & jet_band] = -1.0  # inject to left side
        self.jet_mask[right_sector & jet_band] = 1.0   # inject to right side
        wake_start = cy + self.params.cylinder_radius + 5
        self.wake_mask = np.zeros_like(self.cylinder_mask, dtype=bool)
        self.wake_mask[:, wake_start:] = True
        self.wake_mask[self.cylinder_mask] = False

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.shape
        self.u = self.params.inflow * np.ones((h, w), dtype=np.float32)
        self.v = np.zeros((h, w), dtype=np.float32)
        self.u[self.cylinder_mask] = 0.0
        self.v[self.cylinder_mask] = 0.0

    def _derivatives(self, field: Array) -> Tuple[Array, Array]:
        fx = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) * 0.5
        fy = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) * 0.5
        return fx, fy

    def _apply_boundary_conditions(self) -> None:
        self.u[:, 0] = self.params.inflow
        self.v[:, 0] = 0.0
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        self.u[0, :] = self.u[1, :]
        self.u[-1, :] = self.u[-2, :]
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
        self.u[self.cylinder_mask] = 0.0
        self.v[self.cylinder_mask] = 0.0

    def step(self) -> Dict[str, float]:
        nu, dt = self.params.viscosity, self.params.dt
        lap_u = laplacian(self.u)
        lap_v = laplacian(self.v)
        du_dx, du_dy = self._derivatives(self.u)
        dv_dx, dv_dy = self._derivatives(self.v)
        adv_u = self.u * du_dx + self.v * du_dy
        adv_v = self.u * dv_dx + self.v * dv_dy
        error = np.zeros_like(self.u, dtype=np.float32)
        error[self.wake_mask] = self.params.inflow - self.u[self.wake_mask]
        control = self.controller.step(error)
        jet_delta = control["delta"] * self.jet_mask
        self.u += dt * (nu * lap_u - adv_u) + jet_delta
        self.v += dt * (nu * lap_v - adv_v)
        self._apply_boundary_conditions()
        wake_error = self.params.inflow - self.u[self.wake_mask]
        wake_mse = float(np.mean(wake_error ** 2))
        drag_proxy = float(np.mean(np.abs(self.v[self.wake_mask])))
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
            "wake_mse": wake_mse,
            "drag_proxy": drag_proxy,
            "energy": energy,
        }

    def run(self, steps: int, record_interval: int = 20) -> Dict[str, object]:
        metrics = []
        u_history = []
        v_history = []
        for t in range(steps):
            stats = self.step()
            if (t % record_interval) == 0 or t == steps - 1:
                metrics.append({"step": t, **stats})
                u_history.append(self.u.copy())
                v_history.append(self.v.copy())
        return {
            "metrics": metrics,
            "u_history": np.stack(u_history, axis=0),
            "v_history": np.stack(v_history, axis=0),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        u_history: Array = history["u_history"]
        v_history: Array = history["v_history"]
        raster_path = out_path / "flow_cylinder_velocity.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(u_history[-1], cmap="coolwarm", vmin=0.0, vmax=max(1.0, np.max(u_history[-1])))
        axes[0].set_title("u velocity")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(v_history[-1], cmap="coolwarm", vmin=np.min(v_history[-1]), vmax=np.max(v_history[-1]))
        axes[1].set_title("v velocity")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)
        np.savez_compressed(out_path / "flow_cylinder_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "flow_cylinder_history.npz"),
        }


__all__ = [
    "FlowCylinderParams",
    "FlowCylinderControlConfig",
    "FlowCylinderController",
    "FlowCylinderSimulator",
]
