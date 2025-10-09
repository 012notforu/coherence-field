from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
)
from .heat_diffusion_periodic import (
    HeatPeriodicParams,
    HeatPeriodicSimulator,
    synthetic_periodic_temperature,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCFD heat diffusion benchmark with periodic boundaries")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record-interval", type=int, default=40)
    parser.add_argument("--outdir", type=str, default="heat_periodic_outputs")
    parser.add_argument("--target", type=str, default="stripe", choices=["stripe", "checker", "spiral", "tilted", "mixed"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--dt-jitter", type=float, default=0.0)
    parser.add_argument("--control-budget", type=float, default=12.0, help="Total L1 control budget per step; negative disables")
    parser.add_argument("--phase", type=float, default=0.0, help="Phase offset for periodic patterns")
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    parser.add_argument("--metrics-json", type=str, default="", help="Optional path to dump metrics as JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    budget = None if args.control_budget < 0 else float(args.control_budget)
    params = HeatPeriodicParams(init_seed=args.seed, dt=args.dt, dt_jitter=args.dt_jitter, control_budget=budget)
    control_cfg = HeatDiffusionControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = HeatDiffusionController(control_cfg, params.shape)
    target = synthetic_periodic_temperature(params.shape, kind=args.target, phase=args.phase)
    simulator = HeatPeriodicSimulator(params, controller, target)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_np = np.array(history["metrics"], dtype=object)
    np.save(out_dir / "metrics.npy", metrics_np)

    vis = simulator.generate_visualization(out_dir, history)
    last_metrics = history["metrics"][-1]
    print("Heat periodic metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")

    if args.metrics_json:
        json_path = Path(args.metrics_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"metrics": history["metrics"], "last": last_metrics}, indent=2))


if __name__ == "__main__":
    main()
