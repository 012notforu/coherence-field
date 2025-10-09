"""Heat diffusion variant emphasising periodic boundary conditions."""
from __future__ import annotations

import time
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
    HeatDiffusionParams,
    HeatDiffusionSimulator,
)

Array = np.ndarray


@dataclass
class HeatPeriodicParams(HeatDiffusionParams):
    """Parameters for the periodic heat benchmark."""

    dt_jitter: float = 0.0
    control_budget: float | None = 12.0


class HeatPeriodicSimulator(HeatDiffusionSimulator):
    """Heat diffusion simulator with wrap-around emphasis and richer metrics."""

    def __init__(
        self,
        params: HeatPeriodicParams,
        controller: HeatDiffusionController,
        target: Array,
    ) -> None:
        super().__init__(params, controller, target)
        self.params = params
        self.prev_energy: float | None = None

    def reset(self) -> None:
        super().reset()
        self.prev_energy = None

    def _apply_budget(self, delta: Array) -> tuple[Array, float, float]:
        total_control = float(np.sum(np.abs(delta)))
        if self.params.control_budget is None or total_control <= 0.0:
            return delta, total_control, 1.0
        if total_control <= self.params.control_budget:
            budget_use = total_control / max(self.params.control_budget, 1e-8)
            return delta, total_control, budget_use
        scale = self.params.control_budget / (total_control + 1e-8)
        return delta * scale, total_control, min(1.0, scale * total_control / max(self.params.control_budget, 1e-8))

    def _wrap_error(self) -> float:
        top_bottom = np.mean((self.temp[0, :] - self.temp[-1, :]) ** 2)
        left_right = np.mean((self.temp[:, 0] - self.temp[:, -1]) ** 2)
        return 0.5 * (float(top_bottom) + float(left_right))

    def step(self) -> Dict[str, float]:
        alpha = self.params.alpha
        dt = self.params.dt
        if self.params.dt_jitter > 0.0:
            jitter = self.params.dt_jitter * (self.rng.random() - 0.5)
            dt = max(1e-4, dt + jitter)
        lap = laplacian(self.temp)
        error = self.target - self.temp
        start = time.perf_counter()
        control = self.controller.step(error)
        latency_ms = (time.perf_counter() - start) * 1e3
        raw_delta = control["delta"]
        clip_level = float(self.controller.cfg.control_clip)
        clip_hits = float(np.mean(np.isclose(np.abs(raw_delta), clip_level, atol=1e-5)))
        applied_delta, total_control, budget_util = self._apply_budget(raw_delta)
        self.temp += dt * (alpha * lap + applied_delta)
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
        boundary_wrap = self._wrap_error()
        control_norm = float(np.linalg.norm(applied_delta))
        return {
            "mse": mse,
            "energy": energy,
            "delta_energy": delta_energy,
            "boundary_wrap_mse": boundary_wrap,
            "control_norm": control_norm,
            "control_total_l1": total_control,
            "control_clip_fraction": clip_hits,
            "budget_utilisation": budget_util,
            "controller_latency_ms": float(latency_ms),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        artifacts = super().generate_visualization(out_dir, history)
        out_path = Path(out_dir)
        history_array: Array = history["history"]
        seam_path = out_path / "heat_periodic_wrap_error.png"
        final_state = history_array[-1]
        wrap_error_map = np.abs(final_state - np.roll(final_state, -1, axis=0))
        wrap_error_map += np.abs(final_state - np.roll(final_state, -1, axis=1))
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(wrap_error_map, cmap="magma")
        ax.set_title("Wrap error magnitude")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(seam_path, dpi=160)
        plt.close(fig)
        artifacts["wrap_error"] = str(seam_path)
        return artifacts


def synthetic_periodic_temperature(
    shape: Tuple[int, int],
    kind: str = "stripe",
    phase: float = 0.0,
) -> Array:
    """Generate periodic targets that stitch cleanly across both axes."""

    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h, endpoint=False), np.linspace(0.0, 1.0, w, endpoint=False), indexing="ij")
    if kind == "stripe":
        pattern = 0.5 + 0.45 * np.sin(2.0 * np.pi * (xx + phase))
    elif kind == "checker":
        pattern = 0.5 + 0.4 * np.sin(2.0 * np.pi * (xx + phase)) * np.sin(2.0 * np.pi * (yy + 0.5 * phase))
    elif kind == "spiral":
        radius = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        angle = np.arctan2(yy - 0.5, xx - 0.5)
        pattern = 0.5 + 0.3 * np.sin(6.0 * np.pi * radius + 2.0 * angle + phase)
    elif kind == "tilted":
        pattern = 0.5 + 0.45 * np.sin(2.0 * np.pi * (0.6 * xx + 0.4 * yy + phase))
    else:
        pattern = 0.5 + 0.25 * np.sin(4.0 * np.pi * (xx + yy + phase))
    pattern = pattern.astype(np.float32)
    pattern[-1, :] = pattern[0, :]
    pattern[:, -1] = pattern[:, 0]
    return np.clip(pattern, 0.0, 1.0)


__all__ = [
    "HeatPeriodicParams",
    "HeatPeriodicSimulator",
    "synthetic_periodic_temperature",
    "HeatDiffusionControlConfig",
    "HeatDiffusionController",
]

