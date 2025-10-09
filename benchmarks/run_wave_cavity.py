from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .wave_field_cavity import (
    WaveCavityParams,
    WaveCavityControlConfig,
    WaveCavityController,
    WaveCavitySimulator,
    standing_mode_target,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standing-mode cavity wave benchmark")
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--record-interval", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="wave_cavity_outputs")
    parser.add_argument("--mode-m", type=int, default=2)
    parser.add_argument("--mode-n", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.001)
    parser.add_argument("--wave-speed", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scfd-cfg", type=str, default="cfg/defaults.yaml")
    parser.add_argument("--metrics-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = WaveCavityParams(
        wave_speed=args.wave_speed,
        dt=args.dt,
        damping=args.damping,
        mode_m=args.mode_m,
        mode_n=args.mode_n,
        init_seed=args.seed,
    )
    control_cfg = WaveCavityControlConfig(scfd_cfg_path=args.scfd_cfg)
    controller = WaveCavityController(control_cfg, params.shape)
    target = standing_mode_target(params.shape, params.mode_m, params.mode_n)
    simulator = WaveCavitySimulator(params, controller, target)

    history = simulator.run(steps=args.steps, record_interval=args.record_interval)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "metrics.npy", np.array(history["metrics"], dtype=object))
    vis = simulator.generate_visualization(out_dir, history)

    print("Wave cavity metrics (sampled):")
    for entry in history["metrics"][-5:]:
        print(entry)
    print("Artifacts saved to:")
    for key, value in vis.items():
        print(f"  {key}: {value}")

    if args.metrics_json:
        payload = {
            "metrics": history["metrics"],
            "last": history["metrics"][-1],
        }
        json_path = Path(args.metrics_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
