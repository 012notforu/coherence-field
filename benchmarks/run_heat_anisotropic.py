from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .heat_diffusion import (
    HeatDiffusionControlConfig,
    HeatDiffusionController,
)
from .heat_diffusion_anisotropic import (
    HeatAnisotropicParams,
    HeatAnisotropicSimulator,
    synthetic_anisotropic_temperature,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anisotropic heat diffusion benchmark")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--record-interval", type=int, default=60)
    parser.add_argument("--outdir", type=str, default="heat_anisotropic_outputs")
    parser.add_argument("--target", type=str, default="tilted", choices=["tilted", "elliptic_hotspot", "gradient"])
    parser.add_argument("--angle-deg", type=float, default=22.5, help="Target orientation in degrees")
    parser.add_argument("--orientation-deg", type=float, default=30.0, help="Diffusion tensor orientation (degrees)")
    parser.add_argument("--major", type=float, default=0.24, help="Major diffusion coefficient")
    parser.add_argument("--minor", type=float, default=0.08, help="Minor diffusion coefficient")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    parser.add_argument("--metrics-json", type=str, default="", help="Optional JSON path for metrics dump")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    orientation = np.deg2rad(args.orientation_deg)
    angle = np.deg2rad(args.angle_deg)

    params = HeatAnisotropicParams(
        init_seed=args.seed,
        dt=args.dt,
        alpha_major=args.major,
        alpha_minor=args.minor,
        orientation=float(orientation),
    )
    control_cfg = HeatDiffusionControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = HeatDiffusionController(control_cfg, params.shape)
    target = synthetic_anisotropic_temperature(params.shape, kind=args.target, angle=float(angle))
    simulator = HeatAnisotropicSimulator(params, controller, target)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_np = np.array(history["metrics"], dtype=object)
    np.save(out_dir / "metrics.npy", metrics_np)
    vis = simulator.generate_visualization(out_dir, history)

    print("Heat anisotropic metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")

    if args.metrics_json:
        payload = {
            "metrics": history["metrics"],
            "last": history["metrics"][-1],
            "orientation_deg": args.orientation_deg,
            "tensor": {
                "major": args.major,
                "minor": args.minor,
            },
        }
        json_path = Path(args.metrics_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
