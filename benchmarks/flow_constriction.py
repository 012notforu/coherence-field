"""Flow control benchmark for channel with constriction."""
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
class FlowConstrictionParams:
    shape: Tuple[int, int] = (96, 96)
    viscosity: float = 0.02
    dt: float = 0.05
    inflow: float = 1.0
    slit_height: int = 18
    constriction_half_width: int = 4
    wall_thickness: int = 3
    init_seed: int = 0
    target_velocity: float = 0.9


class FlowConstrictionSimulator:
    """2D channel flow with central constriction and SCFD boundary actuation."""

    def __init__(
        self,
        params: FlowConstrictionParams,
        controller: FlowCylinderController,
    ) -> None:
        self.params = params
        self.controller = controller
        self.shape = params.shape
        self.solid_mask, self.actuator_mask, self.monitor_mask = self._build_geometry()
        self.rng = np.random.default_rng(params.init_seed)
        self.prev_energy: float | None = None
        self.reset()

    def _build_geometry(self) -> tuple[Array, Array, Array]:
        h, w = self.shape
        mask = np.zeros((h, w), dtype=bool)
        wall = max(2, self.params.wall_thickness)
        mask[:wall, :] = True
        mask[-wall:, :] = True
        center = h // 2
        slit_half = max(2, self.params.slit_height // 2)
        half_width = max(2, self.params.constriction_half_width)
        x0 = w // 2 - half_width
        x1 = w // 2 + half_width
        mask[wall:center - slit_half, x0:x1] = True
        mask[center + slit_half:h - wall, x0:x1] = True
        actuator = np.zeros((h, w), dtype=np.float32)
        sleeve = max(2, wall + 1)
        actuator[center - slit_half - sleeve:center - slit_half, x0 - 1:x1 + 1] = 1.0
        actuator[center + slit_half:center + slit_half + sleeve, x0 - 1:x1 + 1] = -1.0
        actuator[mask] = 0.0
        monitor = np.zeros((h, w), dtype=bool)
        monitor[:, -w // 5 :] = True
        monitor[mask] = False
        return mask, actuator, monitor

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.shape
        self.u = self.params.inflow * np.ones((h, w), dtype=np.float32)
        self.v = np.zeros((h, w), dtype=np.float32)
        self.u[self.solid_mask] = 0.0
        self.v[self.solid_mask] = 0.0
        self.prev_energy = None

    def _apply_boundaries(self) -> None:
        self.u[:, 0] = self.params.inflow
        self.v[:, 0] = 0.0
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        self.u[self.solid_mask] = 0.0
        self.v[self.solid_mask] = 0.0

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
        jet_delta = control["delta"] * self.actuator_mask

        self.u += dt * (nu * lap_u - adv_u) + jet_delta
        self.v += dt * (nu * lap_v - adv_v)
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
        delta_energy = 0.0 if self.prev_energy is None else energy - self.prev_energy
        self.prev_energy = energy

        throughput = float(np.mean(self.u[self.monitor_mask]))
        backflow = float(np.mean(np.maximum(0.0, -self.u[self.monitor_mask])))
        control_norm = float(np.linalg.norm(jet_delta))

        return {
            "throughput": throughput,
            "backflow": backflow,
            "energy": energy,
            "delta_energy": delta_energy,
            "control_norm": control_norm,
        }

    def run(self, steps: int, record_interval: int = 20) -> Dict[str, object]:
        metrics = []
        u_hist = []
        for t in range(steps):
            stats = self.step()
            if (t % record_interval) == 0 or t == steps - 1:
                metrics.append({"step": t, **stats})
                u_hist.append(self.u.copy())
        return {
            "metrics": metrics,
            "u_history": np.stack(u_hist, axis=0),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        raster_path = out_path / "flow_constriction_velocity.png"
        mask_path = out_path / "flow_constriction_geometry.png"

        u_hist: Array = history["u_history"]
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(u_hist[-1], cmap="coolwarm")
        ax.set_title("Final u velocity")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.imshow(self.solid_mask, cmap="gray")
        ax2.set_title("Constriction mask")
        ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(mask_path, dpi=160)
        plt.close(fig2)

        np.savez_compressed(out_path / "flow_constriction_history.npz", **history)
        return {
            "raster": str(raster_path),
            "geometry": str(mask_path),
            "history": str(out_path / "flow_constriction_history.npz"),
        }


__all__ = [
    "FlowConstrictionParams",
    "FlowConstrictionSimulator",
]
