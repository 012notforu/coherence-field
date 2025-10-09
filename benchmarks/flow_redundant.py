"""Flow control benchmark with redundant body-force actuators."""
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

from .flow_cylinder import FlowCylinderControlConfig, FlowCylinderController

Array = np.ndarray


@dataclass
class FlowRedundantParams:
    shape: Tuple[int, int] = (96, 96)
    viscosity: float = 0.02
    dt: float = 0.05
    inflow: float = 1.0
    actuator_rows: Tuple[int, ...] = (48,)
    actuator_cols: Tuple[int, ...] = (32, 48, 64)
    actuator_radius: int = 4
    control_budget: float = 3.0
    init_seed: int = 0
    target_velocity: float = 0.9


class FlowRedundantSimulator:
    def __init__(
        self,
        params: FlowRedundantParams,
        controller: FlowCylinderController,
    ) -> None:
        self.params = params
        self.controller = controller
        self.shape = params.shape
        self.rng = np.random.default_rng(params.init_seed)
        self.actuator_masks = self._build_actuators()
        self.monitor_mask = self._build_monitor()
        self._budget_usage = 0.0
        self.reset()

    def _build_actuators(self) -> Tuple[Array, ...]:
        h, w = self.shape
        radius = max(1, self.params.actuator_radius)
        masks = []
        for row in self.params.actuator_rows:
            for col in self.params.actuator_cols:
                mask = np.zeros((h, w), dtype=np.float32)
                yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                dist_sq = (yy - int(row)) ** 2 + (xx - int(col)) ** 2
                mask[dist_sq <= radius ** 2] = 1.0
                if mask.sum() > 0:
                    mask /= float(mask.sum())
                masks.append(mask)
        return tuple(masks)

    def _build_monitor(self) -> Array:
        h, w = self.shape
        monitor = np.zeros((h, w), dtype=bool)
        monitor[:, -w // 4 :] = True
        return monitor

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.shape
        self.u = self.params.inflow * np.ones((h, w), dtype=np.float32)
        self.v = np.zeros((h, w), dtype=np.float32)
        self._budget_usage = 0.0

    @staticmethod
    def _derivatives(field: Array) -> tuple[Array, Array]:
        fx = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) * 0.5
        fy = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) * 0.5
        return fx, fy

    def step(self) -> Dict[str, float]:
        nu, dt = self.params.viscosity, self.params.dt
        lap_u = laplacian(self.u)
        lap_v = laplacian(self.v)
        du_dx, du_dy = self._derivatives(self.u)
        dv_dx, dv_dy = self._derivatives(self.v)
        adv_u = self.u * du_dx + self.v * du_dy
        adv_v = self.u * dv_dx + self.v * dv_dy

        error = np.zeros_like(self.u)
        error[self.monitor_mask] = self.params.target_velocity - self.u[self.monitor_mask]
        control = self.controller.step(error)
        raw = control["delta"]

        amplitudes = []
        for mask in self.actuator_masks:
            weight = float(np.sum(raw * mask))
            amplitudes.append(weight)
        total_l1 = sum(abs(a) for a in amplitudes)
        scale = 1.0
        if self.params.control_budget > 0.0 and total_l1 > 0.0:
            available = max(0.0, self.params.control_budget - self._budget_usage)
            max_allow = available / total_l1 if total_l1 > 1e-6 else 0.0
            scale = min(1.0, max_allow)
            self._budget_usage += min(total_l1, available)
        delta_field = np.zeros_like(self.u)
        for amp, mask in zip(amplitudes, self.actuator_masks):
            delta_field += (amp * scale) * mask

        self.u += dt * (nu * lap_u - adv_u) + delta_field
        self.v += dt * (nu * lap_v - adv_v)

        throughput = float(np.mean(self.u[self.monitor_mask]))
        energy = float(
            np.mean(
                total_energy_density(
                    self.controller.theta,
                    self.controller.theta_dot,
                    self.controller.sim_cfg.physics,
                )
            )
        )
        budget = 0.0 if self.params.control_budget <= 0.0 else min(1.0, self._budget_usage / self.params.control_budget)
        actuator_rms = float(np.sqrt(np.mean(np.square(np.asarray(amplitudes, dtype=np.float32))))) if amplitudes else 0.0
        return {
            "throughput": throughput,
            "energy": energy,
            "budget_util": budget,
            "actuator_rms": actuator_rms,
        }

    def run(self, steps: int, record_interval: int = 40) -> Dict[str, object]:
        metrics = []
        history = []
        for t in range(steps):
            stats = self.step()
            if (t % record_interval) == 0 or t == steps - 1:
                metrics.append({"step": t, **stats})
                history.append(self.u.copy())
        return {
            "metrics": metrics,
            "u_history": np.stack(history, axis=0) if history else np.empty((0,) + self.params.shape),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        raster_path = out_path / "flow_redundant_velocity.png"
        mask_path = out_path / "flow_redundant_actuators.png"

        if "u_history" in history and history["u_history"].size:
            u_hist: Array = history["u_history"]
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(u_hist[-1], cmap="coolwarm")
            ax.set_title("Final u velocity")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(raster_path, dpi=160)
            plt.close(fig)
        else:
            raster_path = str((out_path / "flow_redundant_velocity.png").resolve())

        actuator_stack = np.stack(self.actuator_masks, axis=0) if self.actuator_masks else np.zeros((1,) + self.shape)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.imshow(np.max(actuator_stack, axis=0), cmap="gray")
        ax2.set_title("Actuator coverage")
        ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(mask_path, dpi=160)
        plt.close(fig2)

        np.savez_compressed(out_path / "flow_redundant_history.npz", **history)
        return {
            "raster": str(raster_path),
            "actuators": str(mask_path),
            "history": str(out_path / "flow_redundant_history.npz"),
        }


__all__ = [
    "FlowRedundantParams",
    "FlowRedundantSimulator",
    "FlowCylinderControlConfig",
    "FlowCylinderController",
]
