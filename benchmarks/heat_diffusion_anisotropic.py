"""Heat diffusion benchmark with anisotropic conductivity tensor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import total_energy_density
from engine.ops import grad, divergence

from .heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionParams,
    HeatDiffusionSimulator,
    synthetic_temperature,
)

Array = np.ndarray


@dataclass
class HeatAnisotropicParams(HeatDiffusionParams):
    alpha_major: float = 0.24
    alpha_minor: float = 0.08
    orientation: float = 0.0  # radians


class HeatAnisotropicSimulator(HeatDiffusionSimulator):
    """Heat simulator using an anisotropic diffusion tensor."""

    def __init__(
        self,
        params: HeatAnisotropicParams,
        controller: HeatDiffusionController,
        target: Array,
    ) -> None:
        self.params = params
        self._tensor = self._build_tensor(params)
        self.prev_energy: float | None = None
        super().__init__(params, controller, target)

    @staticmethod
    def _build_tensor(params: HeatAnisotropicParams) -> Array:
        c = float(np.cos(params.orientation))
        s = float(np.sin(params.orientation))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        diag = np.diag([params.alpha_major, params.alpha_minor]).astype(np.float32)
        tensor = rot @ diag @ rot.T
        return tensor.astype(np.float32)

    def reset(self) -> None:
        super().reset()
        self.prev_energy = None

    def _anisotropic_diffusion(self) -> Array:
        gx, gy = grad(self.temp)
        flux_x = self._tensor[0, 0] * gx + self._tensor[0, 1] * gy
        flux_y = self._tensor[1, 0] * gx + self._tensor[1, 1] * gy
        return divergence(flux_x, flux_y)

    def step(self) -> Dict[str, float]:
        dt = self.params.dt
        diff_term = self._anisotropic_diffusion()
        error = self.target - self.temp
        control = self.controller.step(error)
        delta = control["delta"]
        self.temp += dt * (diff_term + delta)

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

        mse = float(np.mean(error ** 2))
        control_norm = float(np.linalg.norm(delta))
        principal_ratio = float(max(self.params.alpha_major, 1e-5) / max(self.params.alpha_minor, 1e-5))

        return {
            "mse": mse,
            "energy": energy,
            "delta_energy": delta_energy,
            "control_norm": control_norm,
            "principal_ratio": principal_ratio,
            "orientation": float(self.params.orientation),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        artifacts = super().generate_visualization(out_dir, history)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        arrow_path = out_path / "heat_anisotropy_axes.png"
        orientation = float(self.params.orientation)
        major = float(self.params.alpha_major)
        minor = float(self.params.alpha_minor)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        c = np.cos(orientation)
        s = np.sin(orientation)
        ax.arrow(0.0, 0.0, c * major, s * major, head_width=0.05, color="tab:red", length_includes_head=True)
        ax.arrow(0.0, 0.0, -c * major, -s * major, head_width=0.05, color="tab:red", length_includes_head=True)
        ax.arrow(0.0, 0.0, -s * minor, c * minor, head_width=0.05, color="tab:blue", length_includes_head=True)
        ax.arrow(0.0, 0.0, s * minor, -c * minor, head_width=0.05, color="tab:blue", length_includes_head=True)
        ax.set_title("Principal diffusion axes")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(arrow_path, dpi=160)
        plt.close(fig)

        artifacts["anisotropy_axes"] = str(arrow_path)
        return artifacts


def synthetic_anisotropic_temperature(
    shape: Tuple[int, int],
    kind: str = "tilted",
    angle: float = 0.3,
) -> Array:
    """Periodic-friendly targets emphasising anisotropic transport."""

    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    if kind == "tilted":
        pattern = 0.5 + 0.4 * np.sin(2.0 * np.pi * (np.cos(angle) * xx + np.sin(angle) * yy))
    elif kind == "elliptic_hotspot":
        rot = angle
        c = np.cos(rot)
        s = np.sin(rot)
        x_rot = c * (xx - 0.6) - s * (yy - 0.4)
        y_rot = s * (xx - 0.6) + c * (yy - 0.4)
        pattern = np.exp(-((x_rot / 0.1) ** 2 + (y_rot / 0.25) ** 2))
    else:
        pattern = synthetic_temperature(shape, kind="gradient")
    return np.clip(pattern.astype(np.float32), 0.0, 1.0)


__all__ = [
    "HeatAnisotropicParams",
    "HeatAnisotropicSimulator",
    "synthetic_anisotropic_temperature",
    "HeatDiffusionControlConfig",
    "HeatDiffusionController",
]
