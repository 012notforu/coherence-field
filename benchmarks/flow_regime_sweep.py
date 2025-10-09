"""Flow control benchmark sweeping multiple inflow regimes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .flow_cylinder import (
    FlowCylinderParams,
    FlowCylinderControlConfig,
    FlowCylinderController,
    FlowCylinderSimulator,
)

Array = np.ndarray


@dataclass
class FlowRegimeParams(FlowCylinderParams):
    inflow_values: Tuple[float, ...] = (0.6, 1.0, 1.4)
    steps_per_regime: int = 800
    record_interval: int = 40


class FlowRegimeSweep:
    """Runs a sequence of inflow regimes using a shared controller."""

    def __init__(
        self,
        params: FlowRegimeParams,
        controller_cfg: FlowCylinderControlConfig,
    ) -> None:
        self.params = params
        self.controller_cfg = controller_cfg
        self.regime_histories: List[Dict[str, object]] = []
        self.regime_metrics: List[Dict[str, float]] = []

    def _run_single(self, inflow: float, seed: int) -> Dict[str, object]:
        base_params = replace(self.params, inflow=inflow, init_seed=seed)
        controller = FlowCylinderController(self.controller_cfg, base_params.shape)
        simulator = FlowCylinderSimulator(base_params, controller)
        history = simulator.run(
            steps=self.params.steps_per_regime,
            record_interval=self.params.record_interval,
        )
        last = history["metrics"][-1]
        metrics = {
            "inflow": float(inflow),
            "wake_mse": float(last["wake_mse"]),
            "drag_proxy": float(last["drag_proxy"]),
            "energy": float(last["energy"]),
        }
        self.regime_histories.append(history)
        self.regime_metrics.append(metrics)
        return history

    def run(self, seeds: Iterable[int]) -> Dict[str, object]:
        self.regime_histories.clear()
        self.regime_metrics.clear()
        seeds = list(seeds)
        if not seeds:
            seeds = [self.params.init_seed]
        for inflow in self.params.inflow_values:
            seed = int(seeds[0])
            seeds = seeds[1:] + [seed]
            self._run_single(inflow, seed)
        return {
            "metrics": self.regime_metrics,
            "histories": self.regime_histories,
        }

    def aggregate_metrics(self) -> Dict[str, float]:
        if not self.regime_metrics:
            raise RuntimeError("run() must be called before aggregate_metrics().")
        wake = np.array([m["wake_mse"] for m in self.regime_metrics], dtype=np.float32)
        drag = np.array([m["drag_proxy"] for m in self.regime_metrics], dtype=np.float32)
        return {
            "mean_wake_mse": float(wake.mean()),
            "std_wake_mse": float(wake.std()),
            "mean_drag_proxy": float(drag.mean()),
        }

    def save_visualizations(self, out_dir: Path) -> Dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        figure_paths: Dict[str, str] = {}
        for idx, (history, metrics) in enumerate(zip(self.regime_histories, self.regime_metrics)):
            inflow = metrics["inflow"]
            raster_path = out_dir / f"flow_regime_{idx}_velocity.png"
            u_history: Array = history["u_history"]
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(u_history[-1], cmap="coolwarm")
            ax.set_title(f"u velocity (inflow={inflow:.2f})")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(raster_path, dpi=160)
            plt.close(fig)
            figure_paths[f"regime_{idx}"] = str(raster_path)
        return figure_paths


__all__ = [
    "FlowRegimeParams",
    "FlowRegimeSweep",
]
