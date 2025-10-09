from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
)
from .heat_diffusion_obstacle import (
    HeatObstacleParams,
    HeatObstacleSimulator,
    synthetic_obstacle_target,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heat diffusion with interior obstacle")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--record-interval", type=int, default=60)
    parser.add_argument("--outdir", type=str, default="heat_obstacle_outputs")
    parser.add_argument("--target", type=str, default="hot_corner", choices=["hot_corner", "cool_corner", "gradient"])
    parser.add_argument("--budget", type=float, default=6.0, help="Per-step L1 control budget (negative disables)")
    parser.add_argument("--gap-height", type=int, default=6)
    parser.add_argument("--gap-width", type=int, default=4)
    parser.add_argument("--obstacle-temp", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    parser.add_argument("--metrics-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    budget = None if args.budget < 0 else float(args.budget)
    params = HeatObstacleParams(
        init_seed=args.seed,
        dt=args.dt,
        control_budget=budget if budget is not None else 0.0,
        gap_height=args.gap_height,
        gap_width=args.gap_width,
        obstacle_temperature=args.obstacle_temp,
    )
    control_cfg = HeatDiffusionControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = HeatDiffusionController(control_cfg, params.shape)
    target = synthetic_obstacle_target(params.shape, kind=args.target)
    simulator = HeatObstacleSimulator(params, controller, target)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_np = np.array(history["metrics"], dtype=object)
    np.save(out_dir / "metrics.npy", metrics_np)
    vis = simulator.generate_visualization(out_dir, history)

    print("Heat obstacle metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")

    if args.metrics_json:
        payload = {
            "metrics": history["metrics"],
            "last": history["metrics"][-1],
            "budget": budget,
            "gap": {
                "height": args.gap_height,
                "width": args.gap_width,
            },
        }
        json_path = Path(args.metrics_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
