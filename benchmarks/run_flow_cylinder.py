from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .flow_cylinder import (
    FlowCylinderParams,
    FlowCylinderControlConfig,
    FlowCylinderController,
    FlowCylinderSimulator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCFD cylinder flow control benchmark")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record-interval", type=int, default=40)
    parser.add_argument("--outdir", type=str, default="flow_cylinder_outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = FlowCylinderParams(init_seed=args.seed)
    control_cfg = FlowCylinderControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = FlowCylinderController(control_cfg, params.shape)
    simulator = FlowCylinderSimulator(params, controller)
    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(out_dir, history)
    print("Flow cylinder metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
