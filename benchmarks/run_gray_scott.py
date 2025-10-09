from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .gray_scott import (
    GrayScottParams,
    GrayScottControlConfig,
    GrayScottController,
    GrayScottSimulator,
    synthetic_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCFD-controlled Gray-Scott benchmark")
    parser.add_argument("--steps", type=int, default=1500, help="Simulation steps")
    parser.add_argument("--record-interval", type=int, default=50, help="History sampling interval")
    parser.add_argument("--outdir", type=str, default="gray_scott_outputs", help="Output directory")
    parser.add_argument("--target", type=str, default="spots", choices=["spots", "stripes", "checker", "waves"], help="Target pattern type")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization")
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml", help="SCFD YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = GrayScottParams(init_seed=args.seed)
    control_cfg = GrayScottControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = GrayScottController(control_cfg, params.shape)
    target = synthetic_target(params.shape, kind=args.target)
    simulator = GrayScottSimulator(params, controller, target_pattern=target)
    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = np.array(history["metrics"], dtype=object)
    np.save(out_dir / "metrics.npy", records)
    vis = simulator.generate_visualization(out_dir, history)
    print("Gray-Scott metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
