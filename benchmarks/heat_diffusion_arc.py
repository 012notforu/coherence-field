"""Heat diffusion benchmark with ARC-style rotate/reflect transforms."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionParams,
    synthetic_temperature,
)
from engine import total_energy_density
from engine.ops import laplacian

Array = np.ndarray

_TRANSFORM_SET = {
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "flip_horizontal",
    "flip_vertical",
    "diag",
    "anti_diag",
}

def apply_arc_transform(field: Array, transform: str) -> Array:
    transform = transform.lower()
    if transform == "identity":
        return field.copy()
    if transform == "rotate90":
        return np.rot90(field, k=1)
    if transform == "rotate180":
        return np.rot90(field, k=2)
    if transform == "rotate270":
        return np.rot90(field, k=3)
    if transform == "flip_horizontal":
        return np.flip(field, axis=1)
    if transform == "flip_vertical":
        return np.flip(field, axis=0)
    if transform == "diag":
        return np.transpose(field)
    if transform == "anti_diag":
        return np.flip(np.transpose(field), axis=1)
    raise ValueError(f"Unknown ARC transform '{transform}'")

def build_transform_cycle(names: Iterable[str]) -> Tuple[str, ...]:
    cycle: list[str] = []
    for name in names:
        clean = name.strip().lower()
        if not clean:
            continue
        if clean not in _TRANSFORM_SET:
            raise ValueError(f"Unknown transform '{clean}'")
        cycle.append(clean)
    if not cycle:
        cycle = ["identity"]
    return tuple(cycle)

@dataclass
class HeatDiffusionArcParams(HeatDiffusionParams):
    transform_cycle: Tuple[str, ...] = field(default_factory=lambda: ("identity", "rotate90", "flip_horizontal"))
    transform_cycle_interval: int = 200
    base_target_kind: str = "gradient"

class HeatDiffusionArcSimulator:
    """Heat diffusion controller tasked with rotating/reflecting target motifs."""

    def __init__(self, params: HeatDiffusionArcParams, controller: HeatDiffusionController) -> None:
        self.params = params
        self.controller = controller
        self.rng = np.random.default_rng(params.init_seed)
        self.shape = params.shape
        self.transform_cycle = params.transform_cycle
        self.transform_interval = max(1, int(params.transform_cycle_interval))
        self.base_target = synthetic_temperature(params.shape, kind=params.base_target_kind)
        self.current_transform_index = 0
        self.target = apply_arc_transform(self.base_target, self.transform_cycle[self.current_transform_index])
        self.steps_since_transform = 0
        self.reset()

    def reset(self) -> None:
        self.controller.reset()
        h, w = self.params.shape
        self.temp = 0.25 * np.ones((h, w), dtype=np.float32)
        self.temp += self.params.noise * self.rng.standard_normal((h, w)).astype(np.float32)
        self.steps_since_transform = 0
        self.current_transform_index = 0
        self.target = apply_arc_transform(self.base_target, self.transform_cycle[self.current_transform_index])

    def _advance_transform(self) -> None:
        self.current_transform_index = (self.current_transform_index + 1) % len(self.transform_cycle)
        self.target = apply_arc_transform(self.base_target, self.transform_cycle[self.current_transform_index])
        self.steps_since_transform = 0

    def step(self) -> Dict[str, float]:
        if self.steps_since_transform >= self.transform_interval:
            self._advance_transform()
        alpha, dt = self.params.alpha, self.params.dt
        lap = laplacian(self.temp)
        error = self.target - self.temp
        control = self.controller.step(error)
        self.temp += dt * (alpha * lap + control["delta"])
        mse = float(np.mean(error**2))
        energy = float(np.mean(total_energy_density(self.controller.theta, self.controller.theta_dot, self.controller.sim_cfg.physics)))
        cycle_error = float(np.mean(np.abs(error)))
        self.steps_since_transform += 1
        return {
            "mse": mse,
            "energy": energy,
            "cycle_error": cycle_error,
        }

    def run(self, steps: int, record_interval: int = 50) -> Dict[str, object]:
        metrics: list[Dict[str, float]] = []
        history: list[Array] = []
        targets: list[Array] = []
        for step in range(steps):
            stats = self.step()
            if (step % record_interval) == 0 or step == steps - 1:
                metrics.append({"step": step, **stats, "transform": self.transform_cycle[self.current_transform_index]})
                history.append(self.temp.copy())
                targets.append(self.target.copy())
        return {
            "metrics": metrics,
            "history": np.stack(history, axis=0) if history else np.empty((0,) + self.params.shape),
            "targets": np.stack(targets, axis=0) if targets else np.empty((0,) + self.params.shape),
        }

    def generate_visualization(self, out_dir: str | Path, history: Dict[str, object]) -> Dict[str, str]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        history_arr: Array = history.get("history", np.empty(0))  # type: ignore[assignment]
        targets_arr: Array = history.get("targets", np.empty(0))  # type: ignore[assignment]

        raster_path = out_path / "heat_arc_composite.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        base_im = axes[0].imshow(self.base_target, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0].set_title("Base pattern")
        fig.colorbar(base_im, ax=axes[0], fraction=0.046, pad=0.04)

        if targets_arr.size:
            tgt_im = axes[1].imshow(targets_arr[-1], cmap="inferno", vmin=0.0, vmax=1.0)
            axes[1].set_title(f"Target ({self.transform_cycle[self.current_transform_index]})")
            fig.colorbar(tgt_im, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            axes[1].imshow(self.target, cmap="inferno", vmin=0.0, vmax=1.0)
            axes[1].set_title("Target")

        if history_arr.size:
            cur_im = axes[2].imshow(history_arr[-1], cmap="inferno", vmin=0.0, vmax=1.0)
        else:
            cur_im = axes[2].imshow(self.temp, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title("Current temperature")
        fig.colorbar(cur_im, ax=axes[2], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(raster_path, dpi=160)
        plt.close(fig)

        np.savez_compressed(out_path / "heat_arc_history.npz", **history)
        return {
            "raster": str(raster_path),
            "history": str(out_path / "heat_arc_history.npz"),
        }

__all__ = [
    "HeatDiffusionArcParams",
    "HeatDiffusionArcSimulator",
    "apply_arc_transform",
    "build_transform_cycle",
]
