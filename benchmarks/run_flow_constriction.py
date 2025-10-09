from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flow_cylinder import FlowCylinderControlConfig, FlowCylinderController
from .flow_constriction import FlowConstrictionParams, FlowConstrictionSimulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flow constriction benchmark")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--record-interval", type=int, default=40)
    parser.add_argument("--outdir", type=str, default="flow_constriction_outputs")
    parser.add_argument("--inflow", type=float, default=1.0)
    parser.add_argument("--target-velocity", type=float, default=0.9)
    parser.add_argument("--slit-height", type=int, default=18)
    parser.add_argument("--half-width", type=int, default=4)
    parser.add_argument("--wall", type=int, default=3)
    parser.add_argument("--viscosity", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    parser.add_argument("--metrics-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = FlowConstrictionParams(
        inflow=args.inflow,
        viscosity=args.viscosity,
        dt=args.dt,
        slit_height=args.slit_height,
        constriction_half_width=args.half_width,
        wall_thickness=args.wall,
        init_seed=args.seed,
        target_velocity=args.target_velocity,
    )
    control_cfg = FlowCylinderControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = FlowCylinderController(control_cfg, params.shape)
    simulator = FlowConstrictionSimulator(params, controller)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_np = np.array(history["metrics"], dtype=object)
    np.save(out_dir / "metrics.npy", metrics_np)
    vis = simulator.generate_visualization(out_dir, history)

    print("Flow constriction metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")

    if args.metrics_json:
        payload = {
            "metrics": history["metrics"],
            "last": history["metrics"][-1],
            "params": {
                "inflow": args.inflow,
                "target_velocity": args.target_velocity,
            },
        }
        json_path = Path(args.metrics_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
