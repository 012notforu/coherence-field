from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .wave_field import (
    WaveFieldParams,
    WaveFieldControlConfig,
    WaveFieldController,
    WaveFieldSimulator,
    synthetic_wave_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCFD wave-field shaping benchmark")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--record-interval", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="wave_field_outputs")
    parser.add_argument("--target", type=str, default="focus", choices=["focus", "defocus", "waves"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = WaveFieldParams(init_seed=args.seed)
    control_cfg = WaveFieldControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = WaveFieldController(control_cfg, params.shape)
    target = synthetic_wave_target(params.shape, kind=args.target)
    simulator = WaveFieldSimulator(params, controller, target)
    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(out_dir, history)
    print("Wave-field metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
