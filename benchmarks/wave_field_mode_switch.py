"""Wave-field benchmark with mid-run target switching."""
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

from .wave_field import (
    WaveFieldControlConfig,
    WaveFieldController,
    WaveFieldParams,
    synthetic_wave_target,
)

Array = np.ndarray


@dataclass
class WaveModeSwitchParams(WaveFieldParams):
    switch_step: int = 600
    initial_kind: str = "focus"
    switch_kind: str = "waves"


class WaveModeSwitchSimulator:
    def __init__(
        self,
        params: WaveModeSwitchParams,
        controller: WaveFieldController,
    ) -> None:
        self.params = params
        self.controller = controller
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
        self._steps = 0
        self.current_target = synthetic_wave_target(self.shape, kind=self.params.initial_kind)

    def _update_target(self) -> None:
        kind = self.params.initial_kind if self._steps < self.params.switch_step else self.params.switch_kind
        self.current_target = synthetic_wave_target(self.shape, kind=kind)

    def step(self) -> Dict[str, float | str]:
        self._update_target()
        c, dt, damping = self.params.wave_speed, self.params.dt, self.params.damping
        lap = laplacian(self.field)
        error = np.zeros_like(self.field)
        error[self.interior_mask] = self.current_target[self.interior_mask] - self.field[self.interior_mask]
        control = self.controller.step(error)
        delta = np.zeros_like(self.field)
        delta[self.boundary_mask] = control["delta"][self.boundary_mask]

        self.velocity += dt * (c ** 2 * lap - damping * self.velocity + delta)
        self.field += dt * self.velocity
        phase = "initial" if self._steps < self.params.switch_step else "switched"
        self._steps += 1

        mse = float(np.mean((self.field - self.current_target) ** 2))
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
            "mse": mse,
            "energy": energy,
            "phase": phase,
        }

    def run(self, steps: int, record_interval: int = 40) -> Dict[str, object]:
        metrics = []
        history = []
        targets = []
        for t in range(steps):
            stats = self.step()
            if (t % record_interval) == 0 or t == steps - 1:
                metrics.append({"step": t, **stats})
                history.append(self.field.copy())
                targets.append(self.current_target.copy())
        return {
            "metrics": metrics,
            "history": np.stack(history, axis=0) if history else np.empty((0,) + self.shape),
            "targets": np.stack(targets, axis=0) if targets else np.empty((0,) + self.shape),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        history_array: Array = history.get("history", np.empty(0))
        target_array: Array = history.get("targets", np.empty(0))
        raster_path = out_path / "wave_mode_switch.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        if target_array.size:
            im0 = axes[0].imshow(target_array[0], cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axes[0].set_title("Initial target")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].imshow(target_array[-1], cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axes[1].set_title("Final target")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            axes[0].imshow(self.current_target, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axes[0].set_title("Initial target")
            axes[1].axis("off")
        if history_array.size:
            im2 = axes[2].imshow(history_array[-1], cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axes[2].set_title("Current field")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        else:
            axes[2].imshow(self.field, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)
        np.savez_compressed(out_path / "wave_mode_switch_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "wave_mode_switch_history.npz"),
        }


__all__ = [
    "WaveModeSwitchParams",
    "WaveModeSwitchSimulator",
    "WaveFieldControlConfig",
    "WaveFieldController",
]
