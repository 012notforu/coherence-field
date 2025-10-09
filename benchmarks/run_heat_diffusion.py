from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .heat_diffusion import (
    HeatDiffusionParams,
    HeatDiffusionControlConfig,
    HeatDiffusionController,
    HeatDiffusionSimulator,
    synthetic_temperature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCFD heat diffusion benchmark")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record-interval", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="heat_diffusion_outputs")
    parser.add_argument("--target", type=str, default="gradient", choices=["gradient", "hot_corner", "cool_spot", "waves"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = HeatDiffusionParams(init_seed=args.seed)
    control_cfg = HeatDiffusionControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = HeatDiffusionController(control_cfg, params.shape)
    target = synthetic_temperature(params.shape, kind=args.target)
    sim = HeatDiffusionSimulator(params, controller, target)
    history = sim.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = sim.generate_visualization(out_dir, history)
    print("Heat diffusion metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
